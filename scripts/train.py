"""Train HeadParsingMobile on processed CelebAMask-HQ data."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the mobile head parsing model.")
    parser.add_argument("--model-config", type=Path, default=Path("configs/model_mobilev2_lraspp_320.yaml"))
    parser.add_argument("--dataset-config", type=Path, default=Path("configs/dataset_celebamask_hq.yaml"))
    parser.add_argument("--train-config", type=Path, default=Path("configs/train_student.yaml"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/train"))
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--disable-teacher", action="store_true", help="Ignore teacher/distillation config.")
    return parser.parse_args()


def require_torch():
    try:
        import torch
    except ImportError as exc:
        raise SystemExit("PyTorch is required for training. Install the server training environment first.") from exc
    return torch


def build_poly_scheduler(torch, optimizer, total_steps: int, warmup_steps: int, power: float):
    def lr_lambda(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return max(1e-6, float(step + 1) / float(warmup_steps))
        progress = min(1.0, float(step - warmup_steps) / float(max(1, total_steps - warmup_steps)))
        return (1.0 - progress) ** power

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


def resolve_project_path(path_value: str | Path | None) -> Path | None:
    if path_value is None:
        return None
    path = Path(path_value)
    if path.is_absolute():
        return path
    return (PROJECT_ROOT / path).resolve()


def teacher_backbone_from_name(name: str) -> str:
    if name in {"resnet18", "resnet34"}:
        return name
    if name.endswith("resnet18"):
        return "resnet18"
    if name.endswith("resnet34"):
        return "resnet34"
    raise ValueError(f"Unsupported teacher model name: {name}")


def build_train_transform_kwargs(train_cfg: dict) -> dict:
    from src.utils.config import get_nested

    kwargs = {}
    scale_range = get_nested(train_cfg, "train.augmentation.random_scale")
    if scale_range is not None:
        kwargs["scale_range"] = tuple(float(value) for value in scale_range)
    rotate = get_nested(train_cfg, "train.augmentation.random_rotate_deg")
    if rotate is not None:
        kwargs["rotate_degrees"] = float(rotate)
    translate = get_nested(train_cfg, "train.augmentation.random_translate_ratio")
    if translate is not None:
        kwargs["translate_ratio"] = float(translate)
    random_letterbox = get_nested(train_cfg, "train.augmentation.random_letterbox")
    if random_letterbox is not None:
        kwargs["random_letterbox"] = bool(random_letterbox)
    color_jitter = get_nested(train_cfg, "train.augmentation.color_jitter")
    if color_jitter is not None:
        kwargs["color_jitter_prob"] = 0.8 if bool(color_jitter) else 0.0
    blur = get_nested(train_cfg, "train.augmentation.blur")
    if blur is not None:
        kwargs["blur_prob"] = 0.2 if bool(blur) else 0.0
    jpeg = get_nested(train_cfg, "train.augmentation.jpeg")
    if jpeg is not None:
        kwargs["jpeg_prob"] = 0.2 if bool(jpeg) else 0.0
    occlusion = get_nested(train_cfg, "train.augmentation.occlusion")
    if occlusion is not None:
        kwargs["occlusion_prob"] = 0.25 if bool(occlusion) else 0.0
    return kwargs


def main() -> None:
    args = parse_args()
    torch = require_torch()

    from torch.utils.data import DataLoader

    from src.datasets import CelebAMaskHQDataset
    from src.models import HeadParsingMobile
    from src.training import (
        CombinedSegmentationLoss,
        SegmentationLossConfig,
        SoftTargetDistillationLoss,
        load_bisenet_teacher,
    )
    from src.training.trainer import evaluate, train_one_epoch
    from src.utils.checkpoint import load_checkpoint, save_checkpoint
    from src.utils.config import get_nested, load_yaml
    from src.utils.seed import seed_everything

    model_cfg = load_yaml(args.model_config)
    dataset_cfg = load_yaml(args.dataset_config)
    train_cfg = load_yaml(args.train_config)

    seed_everything(int(get_nested(train_cfg, "train.seed", 42)))

    requested_device = args.device
    if requested_device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but is not available. Use --device cpu for smoke tests.")
    device = torch.device(requested_device)

    input_size = int(get_nested(model_cfg, "model.input_size", [320, 320])[0])
    num_classes = int(get_nested(model_cfg, "model.num_classes", 20))

    train_dataset = CelebAMaskHQDataset.train(
        image_dir=get_nested(dataset_cfg, "dataset.image_dir"),
        mask_dir=get_nested(dataset_cfg, "dataset.mask_dir"),
        split_file=get_nested(dataset_cfg, "dataset.splits.train"),
        output_size=input_size,
        transform_kwargs=build_train_transform_kwargs(train_cfg),
    )
    val_dataset = CelebAMaskHQDataset.eval(
        image_dir=get_nested(dataset_cfg, "dataset.image_dir"),
        mask_dir=get_nested(dataset_cfg, "dataset.mask_dir"),
        split_file=get_nested(dataset_cfg, "dataset.splits.val"),
        output_size=input_size,
    )

    batch_size = int(get_nested(train_cfg, "train.batch_size", 64))
    num_workers = int(get_nested(train_cfg, "train.num_workers", 8))
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )

    model = HeadParsingMobile(
        num_classes=num_classes,
        width_mult=float(get_nested(model_cfg, "model.encoder.width_mult", 1.0)),
        output_stride=int(get_nested(model_cfg, "model.encoder.output_stride", 16)),
        decoder_channels=int(get_nested(model_cfg, "model.decoder.channels", 128)),
        dropout=float(get_nested(model_cfg, "model.decoder.dropout", 0.0)),
    ).to(device)

    loss_config = SegmentationLossConfig(
        num_classes=num_classes,
        ce_weight=float(get_nested(train_cfg, "train.losses.ce_weight", 1.0)),
        dice_weight=float(get_nested(train_cfg, "train.losses.dice_weight", 0.5)),
        boundary_weight=float(get_nested(train_cfg, "train.losses.boundary_weight", 0.2)),
        ohem=bool(get_nested(train_cfg, "train.losses.ohem", True)),
        min_kept=max(1, batch_size * input_size * input_size // 16),
    )
    criterion = CombinedSegmentationLoss(loss_config)

    teacher_model = None
    distillation_criterion = None
    distillation_weight = 0.0
    teacher_enabled = bool(get_nested(train_cfg, "train.teacher.enabled", False)) and not args.disable_teacher
    if teacher_enabled:
        teacher_checkpoint = resolve_project_path(get_nested(train_cfg, "train.teacher.checkpoint"))
        teacher_repo = resolve_project_path(get_nested(train_cfg, "train.teacher.repo_path", "../face-parsing"))
        teacher_model_name = str(get_nested(train_cfg, "train.teacher.model", "bisenet_resnet34"))
        teacher_model = load_bisenet_teacher(
            repo_path=teacher_repo,
            checkpoint_path=teacher_checkpoint,
            backbone=teacher_backbone_from_name(teacher_model_name),
            num_classes=num_classes,
            device=device,
        )
        distillation_criterion = SoftTargetDistillationLoss(
            temperature=float(get_nested(train_cfg, "train.teacher.temperature", 2.0))
        )
        distillation_weight = float(get_nested(train_cfg, "train.losses.distill_weight", 0.5))
        print(f"Loaded teacher model from {teacher_checkpoint} with distillation_weight={distillation_weight}")

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=float(get_nested(train_cfg, "train.optimizer.lr", 0.02)),
        momentum=float(get_nested(train_cfg, "train.optimizer.momentum", 0.9)),
        weight_decay=float(get_nested(train_cfg, "train.optimizer.weight_decay", 0.00004)),
    )

    epochs = int(get_nested(train_cfg, "train.epochs", 160))
    warmup_epochs = int(get_nested(train_cfg, "train.scheduler.warmup_epochs", 3))
    total_steps = epochs * max(1, len(train_loader))
    warmup_steps = warmup_epochs * max(1, len(train_loader))
    scheduler = build_poly_scheduler(
        torch,
        optimizer,
        total_steps=total_steps,
        warmup_steps=warmup_steps,
        power=float(get_nested(train_cfg, "train.scheduler.power", 0.9)),
    )

    start_epoch = 0
    best_miou = 0.0
    if args.resume is not None:
        checkpoint = load_checkpoint(args.resume, map_location=device)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        start_epoch = int(checkpoint.get("epoch", -1)) + 1
        best_miou = float(checkpoint.get("best_miou", 0.0))

    precision = str(get_nested(train_cfg, "train.precision", "amp"))
    scaler = torch.cuda.amp.GradScaler() if device.type == "cuda" and precision == "amp" else None
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(start_epoch, epochs):
        train_stats = train_one_epoch(
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            data_loader=train_loader,
            device=device,
            epoch=epoch,
            scaler=scaler,
            scheduler=scheduler,
            teacher_model=teacher_model,
            distillation_criterion=distillation_criterion,
            distillation_weight=distillation_weight,
        )
        val_stats = evaluate(
            model=model,
            criterion=criterion,
            data_loader=val_loader,
            device=device,
            num_classes=num_classes,
        )

        print(f"epoch={epoch} train={train_stats} val={val_stats}")
        is_best = val_stats["mean_iou"] > best_miou
        best_miou = max(best_miou, val_stats["mean_iou"])
        state = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "best_miou": best_miou,
            "model_config": model_cfg,
            "dataset_config": dataset_cfg,
            "train_config": train_cfg,
        }
        save_checkpoint(state, args.output_dir / "last.pt")
        if is_best:
            save_checkpoint(state, args.output_dir / "best.pt")


if __name__ == "__main__":
    main()
