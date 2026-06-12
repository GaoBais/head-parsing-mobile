# Mouth2Teeth 9 类模型说明与第三方 App 集成指南

本文档面向第三方 Android/iOS App 集成方，说明 `HeadParsingMobile320Mouth2Teeth9Cls` 模型的输入、输出、类别、预处理、后处理、验证方法和与上一版 20 类模型的差异。

本模型不依赖 MediaPipe Face Landmarker。App 可以直接将相机帧、图片或上游业务裁剪后的 RGB 图像送入模型。若 App 已有自己的 ROI/crop 逻辑，可以继续使用，但必须保证预处理和输出 mask 映射逻辑一致。

## 参考来源

本说明基于以下已验证文档和产物整理：

- `docs/mobile_integration.md`：上一版 20 类模型的 Android/iOS 集成约定。
- `docs/android_tflite_integration.md`：Android TFLite CPU/GPU 接入基线。
- `configs/mobile_metadata_mouth2teeth_9cls.json`：9 类模型的输入、输出、类别和 palette 元数据。
- `reports/mouth2teeth_9cls_full_v1/metrics_summary.md`：9 类 full v1 训练、评估和导出结果。

## 交付文件

第三方集成时至少需要以下文件：

| 文件 | 用途 |
| --- | --- |
| `weights/head_parsing_mobile_320_mouth2teeth_9cls_fp16.tflite` | 9 类 TFLite 模型文件 |
| `configs/mobile_metadata_mouth2teeth_9cls.json` | 类别、颜色、输入输出约定元数据 |
| `docs/mouth2teeth_9cls_model_integration.md` | 本集成说明 |

当前模型文件信息：

| 项目 | 值 |
| --- | --- |
| 模型名 | `HeadParsingMobile320Mouth2Teeth9Cls` |
| 任务 | 人头/人脸语义分割 |
| 文件名 | `head_parsing_mobile_320_mouth2teeth_9cls_fp16.tflite` |
| 文件大小 | `9.94 MB` |
| SHA256 | `51042a4b97aaf48ee7a4b702ca58b0f83db53f1addb90058af795a918a87dca7` |
| 权重精度 | fp16 |
| 输入/输出 dtype | float32 |
| softmax | 不包含 |
| argmax | 不包含 |

## 模型规格

| 项目 | 值 |
| --- | --- |
| 输入尺寸 | `320x320` |
| 输入颜色 | RGB |
| 输入 layout | NCHW |
| 输入 shape | `[1, 3, 320, 320]` |
| 输入 dtype | float32 |
| 输出内容 | logits |
| 输出 layout | NCHW |
| 输出 shape | `[1, 9, 320, 320]` |
| 输出 dtype | float32 |
| 类别数 | 9 |

注意：当前 TFLite smoke test 确认的 input/output 都是 NCHW。第三方 App 首次接入时仍应读取 interpreter 的 input/output details，避免 runtime 或转换版本变化导致 layout 认知错误。

## 类别定义

| ID | Class | 中文含义 | 颜色 RGB |
| ---: | --- | --- | --- |
| 0 | `background` | 背景/未保留类别 | `[0, 0, 0]` |
| 1 | `skin` | 面部皮肤 | `[255, 85, 0]` |
| 2 | `l_brow` | 左眉 | `[255, 170, 0]` |
| 3 | `r_brow` | 右眉 | `[255, 0, 85]` |
| 4 | `mouth` | 嘴内部/口腔区域 | `[85, 0, 255]` |
| 5 | `u_lip` | 上唇 | `[170, 0, 255]` |
| 6 | `l_lip` | 下唇 | `[0, 85, 255]` |
| 7 | `teeth` | 牙齿 | `[0, 255, 255]` |
| 8 | `hair` | 头发 | `[255, 255, 170]` |

`l_brow/r_brow` 沿用训练数据集的命名。若 App 发生镜像预览或前置摄像头左右翻转，需要在业务层确认左右语义是否需要再映射。

## 与上一版 20 类模型的差异

上一版模型为 20 类输出，模型文件通常为：

```text
weights/head_parsing_mobile_320_teeth_fp16.tflite
configs/mobile_metadata_mouth2teeth.json
```

9 类模型保留移动端业务需要的类别，并将其重新编号：

| 9 类 ID | 9 类名称 | 原 20 类 ID | 原 20 类名称 |
| ---: | --- | ---: | --- |
| 0 | `background` | 0 | `background` |
| 1 | `skin` | 1 | `skin` |
| 2 | `l_brow` | 2 | `l_brow` |
| 3 | `r_brow` | 3 | `r_brow` |
| 4 | `mouth` | 11 | `mouth` |
| 5 | `u_lip` | 12 | `u_lip` |
| 6 | `l_lip` | 13 | `l_lip` |
| 7 | `teeth` | 19 | `teeth` |
| 8 | `hair` | 17 | `hair` |

旧版中未保留的类别，例如眼睛、耳朵、鼻子、脖子、衣服、帽子等，在 9 类训练数据中都被合并为 `background`。因此第三方 App 不能复用 20 类模型的 class id、palette 或 metadata。

特别注意：

- 旧版 `teeth` id 是 `19`，新版 `teeth` id 是 `7`。
- 旧版 `hair` id 是 `17`，新版 `hair` id 是 `8`。
- 旧版 `mouth` id 是 `11`，新版 `mouth` id 是 `4`。
- 输出 tensor 从 `[1, 20, 320, 320]` 变为 `[1, 9, 320, 320]`，端侧 argmax 和 mask 映射逻辑必须同步更新。

## 预处理

推荐的基础预处理流程：

1. 从相机帧、图片或业务 ROI 得到 RGB 图像。
2. resize 到 `320x320`。
3. 转为 float32。
4. 按 ImageNet mean/std 归一化。
5. 按 NCHW 写入输入 tensor。

归一化公式：

```text
rgb = rgb / 255.0
normalized = (rgb - mean) / std
mean = [0.485, 0.456, 0.406]
std  = [0.229, 0.224, 0.225]
```

NCHW 写入顺序：

```text
input[0, 0, y, x] = normalized_r
input[0, 1, y, x] = normalized_g
input[0, 2, y, x] = normalized_b
```

不要按 NHWC 写入当前模型，否则推理可以运行但结果会明显错误。

## 输入图像尺寸和 ROI 策略

模型本身只接受 `320x320`。App 需要先把原始图像转换到这个尺寸。

推荐按业务场景选择一种固定策略：

| 场景 | 建议 |
| --- | --- |
| 输入已经是方形人脸/头部图 | 直接 resize 到 `320x320` |
| 输入是竖屏自拍预览 | 可先中心裁剪或业务裁剪为头部 ROI，再 resize |
| 输入需要保留完整画面比例 | 可 letterbox 到 `320x320`，但输出 mask 映射回原图时必须去除 padding |
| 输入中人脸很小 | 建议先做业务 ROI 裁剪，否则语义分割精度会下降 |

如果 App 使用 direct resize，会改变非正方形输入的宽高比例；如果使用 letterbox，则需要记录 padding 区域，并在后处理时把 padding 区域从 mask 中去掉。两种方式都可以，但必须在测试、上线和回归中保持一致。

## 后处理

模型输出是 logits，不包含 softmax，也不包含 argmax。

基础后处理：

```text
mask[y, x] = argmax(logits[0, class, y, x])  // class = 0..8
```

说明：

- argmax 前不需要 softmax；softmax 不会改变最大类别。
- 输出 mask 尺寸为 `320x320`，每个像素值是类别 ID。
- 如果需要显示到原始图像尺寸，class-id mask 必须使用 nearest-neighbor resize。
- 不要对 class-id mask 使用 bilinear resize，否则类别边界会产生非法插值。
- 若 App 做了 crop 或 letterbox，必须按相同几何关系把 `320x320` mask 映射回原图。

## Android 集成要点

Android 端建议先跑通 CPU Interpreter，再启用 GPU delegate。

最低接入流程：

1. 将 `.tflite` 放入 `app/src/main/assets/`。
2. 初始化 LiteRT/TFLite Interpreter。
3. 打印并保存 input/output details。
4. 对一张固定测试图片执行 RGB、resize、归一化、NCHW 写入。
5. 执行推理，得到 logits。
6. 对 class/channel 维度做 argmax。
7. 生成 `320x320` class-id mask。
8. 与本项目 `tools/tflite_smoke_test.py` 的输出做对齐。
9. CPU 结果稳定后，再接入 GPU delegate。

GPU delegate 注意事项：

- GPU delegate 不是唯一运行路径，必须保留 CPU fallback。
- 若某些设备 GPU delegate 初始化失败，应记录设备型号、Android 版本、runtime 版本和错误日志。
- GPU 输出应与 CPU 输出在视觉上接近；如果完全不一致，优先检查输入 layout、RGB/BGR 顺序和归一化。

## iOS 集成要点

当前主要交付产物是 TFLite。iOS App 可以选择接入 TFLite runtime，或后续单独导出 Core ML 版本。

无论使用 TFLite 还是 Core ML，都必须保持：

- RGB 输入。
- `320x320` 输入尺寸。
- ImageNet mean/std 归一化。
- 类别顺序为本文档的 9 类顺序。
- 输出 logits 后在端侧做 argmax。
- class-id mask 使用 nearest-neighbor resize。

如果后续使用 Core ML 版本，必须以实际 `.mlpackage` inspection 结果确认输入输出 layout，不要直接假设与 TFLite 完全一致。

## 本地对齐测试

服务器或本地有 LiteRT/TFLite runtime 时，可使用以下命令检查模型：

```bash
python tools/tflite_smoke_test.py \
  --model weights/head_parsing_mobile_320_mouth2teeth_9cls_fp16.tflite \
  --metadata configs/mobile_metadata_mouth2teeth_9cls.json \
  --image data/processed/mouth2teeth_9cls/images/107.jpg \
  --output-dir outputs/tflite_smoke_mouth2teeth_9cls
```

已验证的 smoke test 结果：

| 项目 | 值 |
| --- | --- |
| runtime | `ai_edge_litert.interpreter` |
| input_shape | `[1, 3, 320, 320]` |
| output_shape | `[1, 9, 320, 320]` |
| num_classes | `9` |
| metadata | `configs/mobile_metadata_mouth2teeth_9cls.json` |

当前 smoke test 使用的 `107.jpg` 预测中 `mouth=0`、`teeth=0`，适合作为链路检查，不适合作为牙齿效果验证。第三方 App 做端侧验收时，应额外选一张明确包含牙齿的图片。

## 模型指标

完整训练报告见：

```text
reports/mouth2teeth_9cls_full_v1/metrics_summary.md
```

核心指标：

| 指标 | val | test |
| --- | ---: | ---: |
| 样本数 | 1500 | 1500 |
| pixel_acc | 0.9556 | 0.9574 |
| mean_acc | 0.8666 | 0.8634 |
| mean_iou | 0.7752 | 0.7722 |

Per-class IoU：

| Class | val IoU | test IoU |
| --- | ---: | ---: |
| `background` | 0.9254 | 0.9287 |
| `skin` | 0.9243 | 0.9256 |
| `l_brow` | 0.7489 | 0.7520 |
| `r_brow` | 0.7384 | 0.7487 |
| `mouth` | 0.4913 | 0.4475 |
| `u_lip` | 0.7761 | 0.7782 |
| `l_lip` | 0.8180 | 0.8171 |
| `teeth` | 0.6467 | 0.6387 |
| `hair` | 0.9073 | 0.9130 |

当前主要弱项是 `mouth/teeth/lip` 的边界混淆。App 侧如果重点依赖牙齿或嘴内部区域，建议准备业务样本做额外验收。

## 第三方 App 验收清单

接入完成后至少检查以下内容：

- 模型文件 SHA256 是否匹配本文档记录。
- App 使用的是 9 类 metadata，不是旧 20 类 metadata。
- interpreter input shape 是 `[1, 3, 320, 320]`。
- interpreter output shape 是 `[1, 9, 320, 320]`。
- 输入图像是 RGB，不是 BGR、RGBA 或 YUV 原始顺序。
- 输入 tensor 按 NCHW 写入。
- 归一化公式为 `((rgb / 255.0) - mean) / std`。
- 输出 logits 后按 class/channel 维度做 argmax。
- class-id mask resize 使用 nearest-neighbor。
- `teeth` 使用新版 class id `7`。
- 使用同一张测试图时，App 输出与 `tools/tflite_smoke_test.py` 输出视觉一致或接近。
- 至少使用一张 teeth-positive 图片确认端侧能输出 `teeth` 像素。
- GPU delegate 启用后，与 CPU 输出保持一致或接近，并保留 CPU fallback。

## 常见问题

### 输出几乎全是 background

优先检查 RGB/BGR 顺序、NCHW/NHWC layout、归一化参数和输入图像是否过小。如果人脸或头部在原图中占比很小，应先做业务 ROI 裁剪。

### Android 或 iOS 输出类别错位

通常是复用了旧 20 类 metadata 或 hardcode class id。9 类模型中 `teeth=7`、`hair=8`、`mouth=4`，不能沿用旧版 ID。

### mask 边缘有异常颜色或类别值

检查是否对 class-id mask 使用了 bilinear resize。类别 mask 只能使用 nearest-neighbor resize。

### teeth 没有预测结果

先确认测试图片中确实有清晰牙齿。若本项目 smoke test 有 `teeth` 而 App 没有，优先检查输入 layout、RGB/BGR 和归一化。如果本项目 smoke test 也没有，需要更换 teeth-positive 验收图。

### GPU delegate 失败

保留 CPU fallback，并记录设备型号、系统版本、runtime 版本和错误日志。GPU delegate 存在设备和 op 支持差异，不能作为唯一运行路径。
