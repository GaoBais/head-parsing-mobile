"""Evaluate HeadParsingMobile checkpoints and save visual samples."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a mobile head parsing checkpoint.")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Checkpoint path, usually outputs/train/best.pt.")
    parser.add_argument("--model-config", type=Path, default=Path("configs/model_mobilev2_lraspp_320.yaml"))
    parser.add_argument("--dataset-config", type=Path, default=Path("configs/dataset_celebamask_hq.yaml"))
    parser.add_argument("--train-config", type=Path, default=Path("configs/train_student.yaml"))
    parser.add_argument("--split", choices=["train", "val", "test"], default="val")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/eval"))
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--save-samples", type=int, default=16, help="Number of visual prediction samples to save.")
    parser.add_argument("--no-loss", action="store_true", help="Skip validation loss computation.")
    return parser.parse_args()


def require_torch():
    try:
        import torch
    except ImportError as exc:
        raise SystemExit("PyTorch is required for evaluation. Run this on the server environment.") from exc
    return torch


def resolve_state_dict(checkpoint: Any) -> dict[str, Any]:
    if isinstance(checkpoint, dict):
        for key in ("model", "state_dict", "model_state_dict"):
            value = checkpoint.get(key)
            if isinstance(value, dict):
                return strip_module_prefix(value)
        if checkpoint and all(hasattr(value, "shape") for value in checkpoint.values()):
            return strip_module_prefix(checkpoint)
    raise ValueError("Could not resolve a model state_dict from checkpoint.")


def strip_module_prefix(state_dict: dict[str, Any]) -> dict[str, Any]:
    if not any(key.startswith("module.") for key in state_dict):
        return state_dict
    return {key.removeprefix("module."): value for key, value in state_dict.items()}


def save_metrics(metrics: dict[str, Any], output_path: Path, class_names: list[str]) -> None:
    metrics = dict(metrics)
    if "per_class_iou" in metrics:
        metrics["per_class"] = {
            class_name: {"iou": metrics["per_class_iou"][index]} for index, class_name in enumerate(class_names)
        }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")


def save_visual_samples(torch, model, data_loader, device, output_dir: Path, max_samples: int) -> None:
    if max_samples <= 0:
        return

    import numpy as np
    from PIL import Image

    from src.utils.visualization import make_prediction_grid

    output_dir.mkdir(parents=True, exist_ok=True)
    model.eval()
    saved = 0

    with torch.no_grad():
        for sample in data_loader:
            images = sample["image"].to(device, non_blocking=True)
            targets = sample["mask"]
            logits = model(images)
            preds = torch.argmax(logits, dim=1).detach().cpu().numpy().astype(np.uint8)
            targets_np = targets.detach().cpu().numpy().astype(np.uint8)
            ids = sample["id"]
            image_paths = sample.get("image_path")

            for index in range(preds.shape[0]):
                if saved >= max_samples:
                    return

                if image_paths is not None:
                    image = Image.open(image_paths[index]).convert("RGB")
                else:
                    image = np.zeros((*preds[index].shape, 3), dtype=np.uint8)

                grid = make_prediction_grid(image, preds[index], targets_np[index])
                grid.save(output_dir / f"{ids[index]}_grid.jpg", quality=95)
                Image.fromarray(preds[index]).save(output_dir / f"{ids[index]}_pred.png")
                saved += 1


def main() -> None:
    args = parse_args()
    torch = require_torch()

    from torch.utils.data import DataLoader

    from src.datasets import CelebAMaskHQDataset
    from src.deploy.palette import CLASS_NAMES
    from src.models import HeadParsingMobile
    from src.training import CombinedSegmentationLoss, SegmentationLossConfig
    from src.training.trainer import evaluate
    from src.utils.config import get_nested, load_yaml

    model_cfg = load_yaml(args.model_config)
    dataset_cfg = load_yaml(args.dataset_config)
    train_cfg = load_yaml(args.train_config)

    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but is not available. Use --device cpu for CPU smoke tests.")
    device = torch.device(args.device)

    input_size = int(get_nested(model_cfg, "model.input_size", [320, 320])[0])
    num_classes = int(get_nested(model_cfg, "model.num_classes", 19))
    batch_size = args.batch_size or int(get_nested(train_cfg, "train.batch_size", 64))
    num_workers = args.num_workers if args.num_workers is not None else int(get_nested(train_cfg, "train.num_workers", 8))

    split_path = get_nested(dataset_cfg, f"dataset.splits.{args.split}")
    dataset = CelebAMaskHQDataset.eval(
        image_dir=get_nested(dataset_cfg, "dataset.image_dir"),
        mask_dir=get_nested(dataset_cfg, "dataset.mask_dir"),
        split_file=split_path,
        output_size=input_size,
        return_paths=True,
    )
    data_loader = DataLoader(
        dataset,
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
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(resolve_state_dict(checkpoint))

    criterion = None
    if not args.no_loss:
        criterion = CombinedSegmentationLoss(
            SegmentationLossConfig(
                num_classes=num_classes,
                ce_weight=float(get_nested(train_cfg, "train.losses.ce_weight", 1.0)),
                dice_weight=float(get_nested(train_cfg, "train.losses.dice_weight", 0.5)),
                boundary_weight=float(get_nested(train_cfg, "train.losses.boundary_weight", 0.2)),
                ohem=bool(get_nested(train_cfg, "train.losses.ohem", True)),
                min_kept=max(1, batch_size * input_size * input_size // 16),
            )
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics = evaluate(
        model=model,
        criterion=criterion,
        data_loader=data_loader,
        device=device,
        num_classes=num_classes,
        include_per_class=True,
    )
    metrics["checkpoint"] = str(args.checkpoint)
    metrics["split"] = args.split
    metrics["num_samples"] = len(dataset)
    save_metrics(metrics, args.output_dir / f"metrics_{args.split}.json", CLASS_NAMES)
    save_visual_samples(torch, model, data_loader, device, args.output_dir / "samples", args.save_samples)
    print(json.dumps({key: metrics[key] for key in ("mean_iou", "pixel_acc", "mean_acc")}, indent=2))


if __name__ == "__main__":
    main()
