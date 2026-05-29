# Device Benchmark Protocol

This protocol defines how to benchmark HeadParsingMobile on Android and iOS after server-side training and export.

## Scope

Benchmark the exported mobile artifacts:

- Android: `.tflite` / LiteRT model
- iOS: `.mlpackage` / Core ML model

The protocol does not require the Linux training server. It starts after model export.

## Required Artifacts

For every benchmark run, keep these files together:

```text
weights/
  head_parsing_mobile_320_fp16.tflite
  HeadParsingMobile320.mlpackage
outputs/export/
  onnx_report.json
  tflite_report.json
  coreml_report.json
benchmarks/runs/<run_id>/
  device_android.csv
  device_ios.csv
  artifact_manifest.json
  notes.md
```

## Device Matrix

Minimum matrix:

| Platform | Device Class | Required |
| --- | --- | --- |
| Android | mid-range phone | yes |
| Android | recent flagship | recommended |
| iOS | recent iPhone | yes |
| iOS | older supported iPhone | recommended |

Record exact OS version, SoC, RAM, app build, runtime backend, and thermal state.

## Test Inputs

Use the same input set for every device:

- 30 static images at target resolution class, with varied lighting and hair/hat/ear cases.
- 3 short videos, 10 seconds each, covering frontal, rotation, and motion blur cases.
- For latency-only tests, feed preloaded frames to avoid camera and disk IO noise.

## Metrics

Latency:

- warmup frames: 30
- measured frames: 300
- report p50, p90, p95, p99, min, max, mean
- report end-to-end frame time separately from model-only inference time

Memory:

- model file size
- runtime peak memory if available
- app process RSS before/after model load

Quality:

- visually inspect saved overlays for at least 30 frames
- log failure cases: hair boundary, hat, ears, neck, side face, motion blur, low light
- if ground truth exists, compute mIoU offline using `scripts/evaluate.py`

Thermal:

- run a 3-minute continuous loop
- record whether latency degrades after 1, 2, and 3 minutes
- record device temperature or OS thermal state when available

## Android Procedure

1. Copy `.tflite` and test inputs to the Android app assets or device storage.
2. Run with LiteRT/TFLite GPU delegate.
3. If GPU delegate fails, record the error and run CPU fallback separately.
4. Use fixed input shape, default `1x3x320x320` or app-native equivalent.
5. Exclude camera capture, image decode, and UI rendering from model-only latency.
6. Record preprocessing and postprocessing time separately.

Required Android backend labels:

- `tflite_gpu`
- `tflite_cpu`
- `tflite_nnapi` only if intentionally tested

## iOS Procedure

1. Add `.mlpackage` to the iOS benchmark app.
2. Run with Core ML `MLComputeUnits.all`.
3. Record separate runs for `all`, `cpuAndGPU`, or `cpuOnly` if needed.
4. Exclude camera capture, image decode, and UI rendering from model-only latency.
5. Record preprocessing and postprocessing time separately.

Required iOS backend labels:

- `coreml_all`
- `coreml_cpu_gpu`
- `coreml_cpu`

## Acceptance Targets

Initial v1 targets:

| Metric | Target |
| --- | --- |
| input size | 320x320 |
| Android mid-range model-only p50 | <= 20 ms |
| Android mid-range model-only p95 | <= 35 ms |
| Recent iPhone model-only p50 | <= 12 ms |
| Recent iPhone model-only p95 | <= 24 ms |
| model size | <= 15 MB |
| no delegate fallback | required |
| stable 3-minute run | required |

Adjust targets only after measuring the actual product device class.

## Failure Case Collection

For each failed sample, save:

```text
failures/
  <case_id>_input.jpg
  <case_id>_mask.png
  <case_id>_overlay.jpg
  <case_id>.json
```

Failure metadata should include:

- device
- backend
- model artifact
- input source
- failure class
- short note

Suggested failure classes:

- `hair_boundary`
- `hat_missing`
- `ear_missing`
- `neck_error`
- `low_light`
- `motion_blur`
- `side_face`
- `background_false_positive`

## Reporting

Create one report per benchmark run:

```text
benchmarks/runs/<YYYYMMDD_model_device>/
  artifact_manifest.json
  android_latency.csv
  ios_latency.csv
  failures/
  notes.md
```

Use `tools/inspect_artifacts.py` before device testing to create `artifact_manifest.json`.
