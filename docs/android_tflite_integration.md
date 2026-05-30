# Android TFLite 集成基线

本文档用于把本地 `tools/tflite_smoke_test.py` 的行为迁移到 Android。第一阶段目标是先跑通 CPU Interpreter，并确保 Android 输出与本地 smoke test 对齐；第二阶段再启用 GPU delegate 做实时性能验证。

## 当前模型约定

| 项目 | 值 |
| --- | --- |
| 模型文件 | `head_parsing_mobile_320_teeth_fp16.tflite` |
| assets 路径 | `app/src/main/assets/head_parsing_mobile_320_teeth_fp16.tflite` |
| 输入 | float32 RGB |
| 输入 shape | `[1, 3, 320, 320]` |
| 输入 layout | NCHW |
| 归一化 | `(rgb / 255.0 - mean) / std` |
| mean | `[0.485, 0.456, 0.406]` |
| std | `[0.229, 0.224, 0.225]` |
| 输出 | logits |
| 输出 shape | `[1, 20, 320, 320]` |
| 后处理 | class/channel 维 argmax |
| `teeth` id | `19` |

## 推荐集成顺序

1. 把 `.tflite` 放入 `app/src/main/assets/`。
2. 先用 CPU Interpreter 跑单张图片，得到 `320x320` class-id mask。
3. 使用同一张图片与本地 smoke test 输出做对比。
4. CPU 输出确认一致后，再接入 GPU delegate。
5. GPU delegate 如果初始化失败或某个 op 不支持，需要保留 CPU fallback。

Google 当前文档推荐 Android 上优先考虑 LiteRT with Google Play services；如果应用需要完全离线打包 runtime，也可以使用 standalone LiteRT。GPU delegate 有 op 支持限制，所以本项目必须保留 CPU 对照路径。

官方参考：

- LiteRT Android quickstart: https://ai.google.dev/edge/litert/android/quickstart
- LiteRT with Google Play services: https://ai.google.dev/edge/litert/android/play_services
- Android GPU delegate with Interpreter API: https://ai.google.dev/edge/litert/android/delegates/gpu
- GPU delegate op support: https://ai.google.dev/edge/litert/performance/gpu

## Gradle 依赖选择

### 方案 A：Google Play services runtime

适合大多数带 Google Play services 的 Android 设备。优点是 app 包体更小，runtime 可以由 Play services 提供和更新。

```kotlin
dependencies {
    implementation("com.google.android.gms:play-services-tflite-java:16.5.0")
    implementation("com.google.android.gms:play-services-tflite-gpu:16.5.0")
}
```

版本号应在接入时按官方文档复核。该方案的初始化 API 与 standalone Interpreter 有差异，但输入预处理和输出 argmax 逻辑完全相同。

### 方案 B：standalone LiteRT/TFLite

适合不依赖 Google Play services，或者希望把 runtime 固定进 app 的场景。

```kotlin
dependencies {
    implementation("com.google.ai.edge.litert:litert:<version>")
    implementation("com.google.ai.edge.litert:litert-gpu:<version>")
    implementation("com.google.ai.edge.litert:litert-gpu-api:<version>")
}
```

本项目提供的 `examples/android/HeadParsingTflite.kt` 使用 Interpreter 风格 API 表达完整输入输出流程。实际接入时可以按 app 选定的 runtime API 调整 interpreter 初始化部分，保留预处理和后处理逻辑。按 2026-05-30 官方文档，standalone LiteRT 需要根据 `CompiledModel` 或 `Interpreter` API 路线选择对应版本；接入时不要直接复制 `<version>`。

## 输入处理要求

Android camera 常见来源是 YUV、RGBA 或 BGRA。进入模型前必须得到 RGB 顺序。

最小基线流程：

1. 将输入帧转换为 `Bitmap` 或 RGB buffer。
2. resize 到 `320x320`。
3. 按 NCHW 写入 float buffer：
   - 先写全部 R channel。
   - 再写全部 G channel。
   - 最后写全部 B channel。
4. 对每个通道执行 ImageNet normalization。

注意：不要按 NHWC 写入当前模型，否则推理可以运行但结果会明显错误。

## 输出处理要求

输出是 logits，不包含 softmax，也不需要 softmax 才能取类别。对 class/channel 维取最大值即可：

```text
mask[y, x] = argmax(logits[class, y, x])
```

显示时如果要恢复到相机预览尺寸，class-id mask 必须使用 nearest-neighbor resize，不能使用 bilinear。

## 对齐检查

Android 首次接入后，至少记录：

- Android interpreter input details。
- Android interpreter output details。
- 单张图片输出的 class histogram。
- `teeth` 像素数量。
- CPU 推理耗时。
- GPU delegate 是否可用。
- GPU 输出与 CPU 输出是否一致或接近。

与本地 smoke test 对齐的基准命令：

```powershell
.\.venv-tflite\Scripts\python.exe tools\tflite_smoke_test.py `
  --model weights\head_parsing_mobile_320_teeth_fp16.tflite `
  --image data\processed\mouth2teeth\images\107.jpg `
  --output-dir outputs\tflite_smoke_local_107 `
  --num-threads 4
```

Android 端应使用同一张 `107.jpg` 做第一次对照。

## 常见问题

### Android 输出只有 background 或类别明显错乱

优先检查输入 layout。当前模型是 NCHW，不是 NHWC。

### Android 输出 mask 边缘发虚

检查是否对 class-id mask 使用了 bilinear resize。mask 只能 nearest-neighbor resize。

### GPU delegate 初始化失败

先保留 CPU 路径；记录设备型号、Android 版本、runtime 版本和失败日志。GPU delegate 存在 op 支持限制，不应把 GPU 作为唯一可运行路径。

### `teeth` 没有预测像素

先用本地 smoke test 的同一张图对比。如果本地有 `teeth`，Android 没有，通常是 RGB/BGR、NCHW/NHWC 或 normalization 出错。如果本地也没有，需要换一张包含清晰牙齿的验证图。
