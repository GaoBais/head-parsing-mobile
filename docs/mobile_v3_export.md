# Mobile v3 I/O Export

The v3 export moves normalization, layout adaptation, and argmax into the model
graph so the GPU/ANE does that work, and the on-device preprocessing degrades to
"ROI crop + bilinear + uint8 interleaved write". The network weights are unchanged:
v3 is an export-time re-packaging of the existing 9-class checkpoint.

Consumer-side counterpart: `beauty_ar_sdk/docs/face_parsing_model/mobile_model_v3_io_redesign.md`.

## Contract

| Item | v2 (current) | v3 primary | v3 comparison variant |
| --- | --- | --- | --- |
| Input layout | NCHW `[1,3,320,320]` fp32 | TFLite: NHWC `[1,256,256,3]` uint8; CoreML: image (CVPixelBuffer) | same |
| Normalization | on-device `((rgb/255)-mean)/std` | in-graph: std folded into the stem conv weights (exact algebra); TFLite carries `x*(1/255)-mean` as MUL+SUB, CoreML applies it via `ImageType(scale=1/255, bias=-mean)` | same |
| Output | fp32 logits `[1,9,H,W]` | argmax in-graph: TFLite uint8 `[1,H,W]`, CoreML int32 `[1,H,W]` (native `reduce_argmax`) | logits: TFLite channel-last `[1,H,W,9]`, CoreML `[1,9,H,W]` |
| Input size | 320 | 256 (accuracy-gated, see below) | same |
| Classes | 9, by-name mapping | unchanged | unchanged |

Why a logits comparison variant: TFLite `ARG_MAX` is not in the GPU delegate op
set, so the graph tail may fall to a CPU partition on Android. The primary and
comparison variants share the same weights; the on-device A/B decides which ships
on Android. iOS always uses the primary (argmax) variant.

## Code map

- `src/export/mobile_io.py` — `fold_std_into_stem` (exact algebra, original model
  untouched), `Uint8NormalizedInput`, `ArgmaxHead`, per-framework wrapper builders.
- `src/export/tflite.py` — `TFLiteV3ExportConfig` / `export_tflite_v3`
  (requires a converter with `to_channel_last_io`).
- `src/export/coreml.py` — `CoreMLV3ExportConfig` / `export_coreml_v3`
  (`ImageType` input; `color_layout` RGB or BGR decides the bias order).
- `src/export/metadata.py` — `build_v3_metadata`: v2 structure plus flat
  `input_size` / `input_layout` / `input_dtype` / `output_kind` keys the SDK reads
  to drive its pre/post-processing (replaces hard-coded constants).
- `scripts/export_mobile_v3.py` — orchestrates the four artifacts listed in
  `configs/export_mobile_v3.yaml`, writes a sibling `<stem>_metadata.json` per
  artifact and `outputs/export/v3_export_report.json`.
- `tools/compare_v3_artifacts.py` — the export gate (below).
- `configs/model_mobilev2_lraspp_256_9cls.yaml` — 256 model config for evaluation.

## Runbook (export environment)

```bash
# 1. Export (Linux server: tflite; macOS: coreml; both if the env has both stacks)
python scripts/export_mobile_v3.py --checkpoint outputs/train/best.pt --formats tflite
python scripts/export_mobile_v3.py --checkpoint outputs/train/best.pt --formats coreml

# 2. Agreement gate: >= 99% pixelwise label-map agreement vs the fp32 reference
#    on a fixed val face crop (guards channel-order / normalization-folding bugs).
python tools/compare_v3_artifacts.py --checkpoint outputs/train/best.pt \
    --image <fixed val face crop> --formats tflite

# 3. 256 accuracy gate: evaluate the 320-trained checkpoint at 256 directly.
python scripts/evaluate.py --checkpoint outputs/train/best.pt \
    --model-config configs/model_mobilev2_lraspp_256_9cls.yaml \
    --dataset-config configs/dataset_mouth2teeth_9cls.yaml \
    --train-config configs/train_mouth2teeth_9cls.yaml --split val
```

## Acceptance gates

- **256 direct inference vs 320**: `mean_iou` drop > 2pt, or `mouth` / `teeth`
  per-class IoU drop > 5pt → fine-tune @256 (short schedule, low LR, same data;
  the BiSeNet teacher distillation path in `train.py` can help the weak classes),
  then re-evaluate. Still failing → keep v3 at 320 (the I/O contract benefits
  remain; pre/readback pixel volume stays at the 320 budget).
- **Folding correctness**: folded vs explicit normalization must be ≈0 different
  on val per-class IoU (covered numerically by unit tests; a > 0.1pt gap means a
  folding bug).
- **Artifact agreement**: every exported artifact >= 99% label-map agreement with
  the fp32 reference on the fixed input (`tools/compare_v3_artifacts.py`, writes
  `outputs/export/v3_gate_report.json`).
- **Visual smoke**: overlay masks per class on the sample gallery as in the
  existing mobile integration regression checks.

## Deliverables to the SDK

Four artifacts + their sibling metadata json files:

1. `weights/head_parsing_mobile_256_mouth2teeth_9cls_v3_labelmap_fp16.tflite` (primary Android)
2. `weights/head_parsing_mobile_256_mouth2teeth_9cls_v3_logits_fp16.tflite` (Android A/B)
3. `weights/HeadParsingMobile256Mouth2Teeth9ClsV3.mlpackage` (primary iOS)
4. `weights/HeadParsingMobile256Mouth2Teeth9ClsV3Logits.mlpackage` (comparison)

plus the val evaluation table (320 vs 256, mean and per-class IoU) for the
acceptance decision.

## 256 fine-tune

Use this path when the 320 checkpoint's direct 256 evaluation misses the gate.
`--init-checkpoint` loads only model weights and starts a fresh optimizer/scheduler;
do not use `--resume` for this because `--resume` is for interrupted same-run
training.

```bash
python scripts/train.py \
    --init-checkpoint outputs/train_mouth2teeth_9cls_full_v1/best.pt \
    --model-config configs/model_mobilev2_lraspp_256_9cls.yaml \
    --dataset-config configs/dataset_mouth2teeth_9cls.yaml \
    --train-config configs/train_mouth2teeth_9cls_finetune_256.yaml \
    --output-dir outputs/train_mouth2teeth_9cls_256_ft_v1 \
    --device cuda

python scripts/evaluate.py \
    --checkpoint outputs/train_mouth2teeth_9cls_256_ft_v1/best.pt \
    --model-config configs/model_mobilev2_lraspp_256_9cls.yaml \
    --dataset-config configs/dataset_mouth2teeth_9cls.yaml \
    --train-config configs/train_mouth2teeth_9cls_finetune_256.yaml \
    --split val --device cuda \
    --output-dir outputs/eval_256_v3_ft
```

If the fine-tuned checkpoint passes the 256 gate, re-run v3 export with that
checkpoint so the SDK receives artifacts built from the accepted weights.
