# Implementation Progress

This file is the project-level progress record. Update it whenever an implementation task moves state.

Status values:

- `pending`: not started
- `in_progress`: actively being implemented
- `done`: implemented and locally verified
- `blocked`: cannot continue without external input or missing dependency/data

## Task Summary

Total tasks: 15

| ID | Task | Status | Progress | Deliverable |
| --- | --- | --- | --- | --- |
| T01 | Project scaffold and progress tracking | done | 100% | docs, configs, package skeleton |
| T02 | CelebAMask-HQ preprocessing | done | 100% | mask merge script, split generation |
| T03 | Dataset loader and mobile augmentations | done | 100% | PyTorch dataset and transforms |
| T04 | Mobile model implementation | in_progress | 80% | MobileNetV2 + LR-ASPP-lite |
| T05 | Losses and metrics | in_progress | 80% | CE, Dice, boundary, OHEM, mIoU |
| T06 | Training pipeline | in_progress | 65% | CUDA/AMP trainer and checkpoints |
| T07 | Teacher model and distillation | in_progress | 80% | BiSeNet teacher adapter |
| T08 | Evaluation and visualization | in_progress | 80% | reports and visual samples |
| T09 | ONNX export and graph check | in_progress | 80% | ONNX export smoke test |
| T10 | LiteRT/TFLite export | in_progress | 70% | FP16 TFLite model |
| T11 | Core ML export | in_progress | 70% | FP16 mlpackage |
| T12 | Device benchmark protocol | done | 100% | Android/iOS benchmark checklist |
| T13 | Mouth2teeth class extension | done | 100% | 20-class configs and preprocessing support |
| T14 | Mouth2teeth evaluation report | done | 100% | metrics summary and visual report |
| T15 | TFLite smoke test and mobile integration guide | done | 100% | smoke-test tool, metadata, integration doc |

## Current Notes

- The project is intentionally independent from `../face-parsing`.
- `../face-parsing` remains useful as a teacher/baseline reference.
- Local Windows is the development environment only. It has host Python 3.13.7 at `C:\Users\hengx\AppData\Local\Programs\Python\Python313\python.exe`, but no NVIDIA GPU is expected. Use host Python for local checks; run full CUDA training later on the Linux server.
- Local PyTorch is not installed. Non-PyTorch tests run locally; PyTorch/ONNX-dependent model/loss/distillation/export tests are present but skipped until PyTorch is available on the server or local CPU environment.
- TFLite smoke-test code and docs are locally verified, but this Windows environment does not have a TFLite/LiteRT Python runtime. Run the actual `.tflite` inference smoke test in the server export environment or on device.

## Change Log

- 2026-05-29: Created project scaffold, plan documents, config drafts, and this progress tracker.
- 2026-05-29: Added initial Python package skeleton, class palette, seed helper, requirements, and CelebAMask-HQ preprocessing script.
- 2026-05-29: Verified local host Python 3.13.7 is available; made tqdm optional for lightweight preprocessing checks.
- 2026-05-29: Added `AGENTS.md` with Windows local development and Linux CUDA server training rules; verified host Python 3.13.7 is available.
- 2026-05-29: Implemented processed CelebAMask-HQ dataset reader, mobile training/eval transforms, and unit tests. Local `python -m unittest discover -s tests` passed with 1 PyTorch-dependent model test skipped.
- 2026-05-29: Implemented MobileNetV2 encoder, LR-ASPP-lite decoder, and `HeadParsingMobile`; static Python compilation passed. Runtime shape verification is pending PyTorch.
- 2026-05-29: Implemented CE/OHEM, Dice, boundary loss, segmentation metrics, trainer loops, and `scripts/train.py`. Local `python scripts/train.py --help`, static compilation, and unit tests passed; PyTorch runtime tests are pending server environment.
- 2026-05-29: Implemented BiSeNet teacher adapter, soft-target distillation loss, trainer integration, `--disable-teacher` smoke-test switch, and distillation tests. Local unit tests pass with 3 PyTorch-dependent tests skipped.
- 2026-05-29: Implemented evaluation entrypoint, metrics JSON export, prediction sample rendering, mask visualization utilities, and visualization tests. Local `evaluate.py --help`, `visualize_masks.py --help`, static compilation, and unit tests passed; checkpoint evaluation runtime is pending PyTorch/server.
- 2026-05-29: Implemented ONNX export helper, export CLI, ONNX mobile-op graph checker, and export tests. Local `export_onnx.py --help`, static compilation, and unit tests passed; ONNX runtime export/check is pending server environment.
- 2026-05-29: Implemented LiteRT/TFLite export helper and CLI using `litert_torch` with `ai_edge_torch` fallback, added server requirements, and added TFLite export tests. Local `export_tflite.py --help`, static compilation, and unit tests passed; actual TFLite conversion is pending Linux server environment.
- 2026-05-29: Implemented Core ML export helper and CLI using TorchScript trace plus `coremltools.convert(..., convert_to="mlprogram")`, added Core ML tests, and documented `.mlpackage` export. Local `export_coreml.py --help`, static compilation, and unit tests passed; actual Core ML conversion/loading is pending server or macOS/iOS environment.
- 2026-05-29: Completed device benchmark protocol, Android/iOS latency templates, failure-case template, notes template, and artifact inspection tool. Local CLI help and unit tests passed.
- 2026-05-29: Fixed review findings: disabled unintended teacher backbone default-weight downloads in both `models.resnet` and imported `models.bisenet` references, kept TFLite conversion tracing in FP32 while preserving FP16 target config, aligned export checkpoint defaults with training output, wired augmentation config into training transforms, set the mobile model config to avoid unimplemented pretrained loading, and made `random_letterbox` explicit. Local `python -m unittest discover -s tests`, CLI help checks, and static compilation passed.
- 2026-05-30: Added `teeth` as class ID 19 while preserving the original CelebAMask-HQ class order, updated 20-class model/dataset/training configs, disabled default teacher distillation for the 20-class student, added mouth2teeth dataset/training configs, and made preprocessing ignore zero-area attribute masks.
- 2026-05-30: Hardened TFLite export converter loading: `litert_torch` import compatibility failures now fall back to `ai_edge_torch` in `auto` mode, and `scripts/export_tflite.py` accepts `--converter auto|litert_torch|ai_edge_torch`.
- 2026-05-30: Documented the server training command with `PYTHONUNBUFFERED=1`, `python -u`, and `tee outputs/<run_id>/train.log` so future runs preserve logs required for training-curve reports.
- 2026-05-30: Added a mouth2teeth evaluation report with per-class IoU chart, focused mouth/teeth/lip chart, sample gallery, and artifact manifest.
- 2026-05-30: Completed T15 code/documentation by adding TFLite smoke-test tooling, 20-class mobile metadata, README entry points, and Android/iOS integration guidance. Local `python -m unittest discover -s tests`, static compilation, CLI help, and metadata JSON checks passed; full `.tflite` inference requires a LiteRT/TFLite runtime on server or device.
