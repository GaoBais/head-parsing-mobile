# 移动端集成说明

本文档描述 `HeadParsingMobile320Teeth` 的 Android/iOS 集成约定。当前已验证的移动端优先产物是：

```text
weights/head_parsing_mobile_320_teeth_fp16.tflite
```

配套元数据：

```text
configs/mobile_metadata_mouth2teeth.json
```

## 模型规格

| 项目 | 值 |
| --- | --- |
| 任务 | 人头/人脸语义分割 |
| 输入尺寸 | `320x320` |
| 输入颜色 | RGB |
| 输入布局 | NCHW，`[1, 3, 320, 320]` |
| 输入 dtype | float32 |
| 输出 | logits |
| 输出布局 | NCHW，`[1, 20, 320, 320]` |
| softmax | 未包含 |
| argmax | 未包含 |
| 类别数 | 20 |
| `teeth` class id | 19 |

实际端侧集成前，应使用 `tools/tflite_smoke_test.py` 或端侧 interpreter 再确认一次 TFLite input/output details。如果 runtime 报告 NHWC，则按实际 details 适配输入和输出 layout。

## 预处理

1. 从相机帧或图片得到 RGB。
2. resize 到 `320x320`，使用 bilinear。
3. 转 float32，按 ImageNet 归一化：

```text
rgb = rgb / 255.0
normalized = (rgb - mean) / std
mean = [0.485, 0.456, 0.406]
std  = [0.229, 0.224, 0.225]
```

4. 按模型 input details 填充 tensor。当前导出约定为 NCHW：

```text
[1, 3, 320, 320]
```

## 后处理

1. 输出是 logits，不需要先 softmax 才能取类别。
2. 对 class/channel 维做 argmax。
3. 得到 `320x320` class-id mask。
4. 如需叠加到原图，用 nearest-neighbor 将 mask resize 回显示尺寸。
5. 按 `configs/mobile_metadata_mouth2teeth.json` 中的 palette 做颜色映射。

## 类别顺序

原始 CelebAMask-HQ 19 类顺序不变，`teeth` 追加到最后。

| ID | Class |
| --- | --- |
| 0 | background |
| 1 | skin |
| 2 | l_brow |
| 3 | r_brow |
| 4 | l_eye |
| 5 | r_eye |
| 6 | eye_g |
| 7 | l_ear |
| 8 | r_ear |
| 9 | ear_r |
| 10 | nose |
| 11 | mouth |
| 12 | u_lip |
| 13 | l_lip |
| 14 | neck |
| 15 | neck_l |
| 16 | cloth |
| 17 | hair |
| 18 | hat |
| 19 | teeth |

## 服务器 smoke test

在服务器或有 TFLite runtime 的机器上执行。优先使用已经完成 TFLite 导出的 `head-parsing-export` 环境：

```bash
cd /home/vtai/hengxing/head_parsing
conda activate head-parsing-export

python tools/tflite_smoke_test.py \
  --model weights/head_parsing_mobile_320_teeth_fp16.tflite \
  --image outputs/eval_mouth2teeth_v2/samples/107_grid.jpg \
  --output-dir outputs/tflite_smoke_mouth2teeth_v2
```

更推荐使用原始验证图片，而不是 grid 图。例如：

```bash
python tools/tflite_smoke_test.py \
  --model weights/head_parsing_mobile_320_teeth_fp16.tflite \
  --image data/processed/mouth2teeth/images/107.jpg \
  --output-dir outputs/tflite_smoke_mouth2teeth_v2
```

输出文件：

```text
outputs/tflite_smoke_mouth2teeth_v2/
  tflite_smoke_report.json
  <image>_pred.png
  <image>_color.png
  <image>_overlay.jpg
```

如果只想检查 input/output details，可以不传 `--image`：

```bash
python tools/tflite_smoke_test.py \
  --model weights/head_parsing_mobile_320_teeth_fp16.tflite \
  --output-dir outputs/tflite_smoke_mouth2teeth_v2
```

## Android 集成要点

- 优先使用 TFLite GPU delegate。
- 如果 GPU delegate 不支持某个 op，记录错误并测试 CPU fallback。
- 模型输出是 logits，端侧需要自行 argmax。
- 不要用 bilinear resize class-id mask，应使用 nearest-neighbor。
- 若 Android 图像管线默认是 RGBA/BGRA，需要先转 RGB，并确认通道顺序。

## iOS 集成要点

- 当前主要交付产物是 TFLite；Core ML 可在后续单独导出和验证。
- 如果使用 Core ML，仍需保持同样的 RGB、归一化、类别顺序和 argmax 逻辑。
- Core ML 输出 layout 需要以实际 `.mlpackage` inspection 为准。

## 回归检查

每次替换模型后至少检查：

- input/output tensor details 是否变化。
- `teeth` class id 是否仍为 19。
- 一张包含牙齿的样本是否能预测出 `teeth` 像素。
- Android 端 GPU delegate 是否正常初始化。
- 端侧 mask 与服务器 smoke test 输出是否视觉一致。
