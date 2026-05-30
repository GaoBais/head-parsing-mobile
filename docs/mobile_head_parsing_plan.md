# 移动端实时 Head Parsing 方案

## 1. 背景与目标

目标是在 Android 和 iOS 端实现实时人脸/头部语义分割，不依赖 MediaPipe Face Landmarker。模型直接接收相机帧或经过统一 resize/letterbox 的 RGB 图像，输出头部相关语义类别 mask。

核心约束：

- Linux + CUDA 服务器训练。
- Android 使用 LiteRT/TFLite，优先 GPU delegate。
- iOS 使用 Core ML，优先 Neural Engine/GPU。
- 模型结构尽量只使用移动端稳定算子。
- 训练数据以 CelebAMask-HQ 为主。
- 当前 `../face-parsing` 项目可作为 baseline 或 teacher，不作为最终移动端模型。

## 2. 方案结论

推荐训练一个独立的移动端 student 模型：

```text
HeadParsingMobile
Input: 320x320 RGB
Encoder: MobileNetV2 1.0 or MobileNetV3-Large 0.75
Decoder: LR-ASPP-lite or Fast-SCNN style lightweight decoder
Output: 20-class logits
Deployment: FP16 LiteRT/TFLite + FP16 Core ML
```

首版优先选型：

```text
MobileNetV2 1.0 + LR-ASPP-lite + 320x320
```

理由：

- MobileNetV2 算子更保守，端侧兼容性强。
- Depthwise separable convolution 适合移动端 GPU/NPU。
- 320x320 在精度和实时性之间比较均衡。
- 不依赖 landmark，端侧流程更短，工程集成更简单。

## 3. 与当前 face-parsing 项目的关系

当前项目 `../face-parsing` 使用：

```text
BiSeNet + ResNet18/ResNet34 + 448/512 input
```

它适合作为：

- 高精度 baseline。
- teacher model，用于蒸馏移动端 student。
- 数据处理和可视化参考。

注意：原始 19 类 teacher checkpoint 不能直接蒸馏追加 `teeth` 后的 20 类 student。训练 mouth2teeth 子集时应关闭 teacher，或先准备同样 20 类输出的 teacher。

它不适合作为最终移动端实时主模型，原因是 ResNet backbone 参数量和计算量偏大，且不是移动端算子最优结构。

## 4. 数据集设计

主数据集：

- CelebAMask-HQ
- 约 30,000 张高质量人脸图像
- 默认 CelebAMask-HQ 为 19 类 face/head parsing 标签；mouth2teeth 子集追加 teeth 后为 20 类
- 原始标签是按属性拆分的二值 mask，需要合成为单张 class-id mask

当前项目保留原始 19 类顺序，并将 `teeth` 追加为第 20 类：

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

## 5. 数据预处理

离线预处理：

```text
CelebA-HQ-img/
CelebAMask-HQ-mask-anno/
  -> merge attribute masks
  -> data/processed/celebamask_hq/images/
  -> data/processed/celebamask_hq/masks/
  -> data/splits/train.txt
  -> data/splits/val.txt
  -> data/splits/test.txt
```

mask 合成规则：

- 每个属性二值 mask 对应一个 class id。
- 背景为 0。
- 合成 mask 使用 `uint8` PNG 保存。
- image 和 mask 文件名保持一致。

## 6. 不依赖 Landmarker 时的关键增强

CelebAMask-HQ 是对齐裁剪过的人脸数据。如果端上直接处理相机画面，训练时必须模拟真实输入分布。

必做增强：

- 随机缩放：让头部占输入高度约 `35% - 100%`。
- 随机平移：避免模型只学到居中人脸。
- 随机旋转：建议 `-30` 到 `30` 度。
- 随机 padding/letterbox：模拟不同画幅。
- 随机裁切：模拟头发顶部、下巴、耳朵局部被裁。
- 背景扰动或背景替换：降低对 CelebAMask-HQ 背景分布的依赖。
- 颜色扰动：亮度、对比度、饱和度、白平衡。
- 质量退化：高斯模糊、运动模糊、噪声、JPEG 压缩。
- 遮挡增强：眼镜、口罩、手部、发丝、局部矩形遮挡。

训练时输入生成建议：

```text
原始 512x512 image/mask
  -> random scale/rotate/translate
  -> paste onto random or augmented background
  -> random crop or letterbox to square
  -> resize to 320x320
  -> normalize
```

## 7. 模型结构

### 7.1 Encoder

首版优先：

```text
MobileNetV2 1.0
```

备选：

```text
MobileNetV3-Large 0.75
MobileNetV3-Small 1.0
```

选择原则：

- Android/iOS 兼容性优先时选 MobileNetV2。
- 精度优先且转换验证通过后选 MobileNetV3。

### 7.2 Decoder

推荐 `LR-ASPP-lite`：

```text
low-level feature 1/4 or 1/8
high-level feature 1/16 or 1/32
global average pooling branch
1x1 conv projection
bilinear upsample
concat/add fusion
final 1x1 classifier
```

移动端友好算子：

- `Conv2D`
- `DepthwiseConv2D`
- `BatchNorm` folded into conv
- `ReLU` or `ReLU6`
- `Add`
- `Concat`
- `AveragePool`
- `ResizeBilinear`
- `Softmax` if needed

避免使用：

- dynamic shape control flow
- `grid_sample`
- deformable convolution
- transformer attention
- GroupNorm/LayerNorm
- complex postprocess inside model
- model-side `argmax`

## 8. Loss 与训练策略

推荐总损失：

```text
Loss = CE + Dice + Boundary + Distillation
```

说明：

- `CrossEntropyLoss`：主监督。
- `Dice/Lovasz`：提升小区域和不均衡类别。
- `Boundary loss`：改善头发、嘴唇、眼睛等边界。
- `OHEM`：继续用于困难像素采样。
- `Distillation`：用 BiSeNet teacher 的 soft logits 提升 student 精度。

训练阶段：

1. 数据预处理和 split 固化。
2. 训练或下载 teacher checkpoint。
3. 训练 student baseline，不蒸馏。
4. 加入 teacher distillation。
5. 做 FP16 导出前评估。
6. 做端上 benchmark，根据 latency 调整输入尺寸或 width multiplier。

服务器首次 smoke test 可以先使用 `--disable-teacher` 跳过 teacher checkpoint，确认 student 数据、模型、loss、optimizer 链路跑通后，再启用 BiSeNet teacher 蒸馏。

## 9. 评估指标

离线指标：

- mIoU
- mean pixel accuracy
- per-class IoU
- boundary F-score
- small class IoU: eye, brow, lip, ear, hat

工程指标：

- Android GPU latency
- iOS Core ML latency
- 模型文件大小
- 峰值内存
- 首帧加载耗时
- 连续视频 mask 抖动程度

验收建议：

```text
320x320 FP16 model
Android mid-range GPU: target <= 20 ms/model inference
iPhone recent devices: target <= 12 ms/model inference
model size: target <= 15 MB
```

实际阈值需要以目标设备实测为准。

## 10. 导出与部署

推荐导出链路：

```text
PyTorch checkpoint
  -> ONNX for inspection
  -> TFLite/LiteRT FP16 for Android
  -> Core ML mlpackage FP16 for iOS
```

Android：

- 首版使用 FP16 权重量化。
- 优先 LiteRT/TFLite GPU delegate。
- INT8 作为 CPU/NPU 备选，不作为首版唯一目标。
- PyTorch 到 LiteRT/TFLite 导出优先使用 Google AI Edge 的 LiteRT Torch converter；代码同时兼容 `litert_torch` 和旧包名 `ai_edge_torch`。

iOS：

- 使用 Core ML `mlprogram`。
- Compute units 使用 `ALL`。
- 保持输入输出 shape 固定，减少运行时调度成本。
- Core ML 导出使用 Core ML Tools 的 PyTorch TorchScript trace 转换路径，输出 `.mlpackage`。

端侧流程：

```text
camera frame
  -> resize or letterbox to 320x320
  -> model inference
  -> mask resize back to original frame
  -> optional temporal smoothing
  -> optional shader blend / matting refinement
```

## 11. 风险与缓解

风险 1：CelebAMask-HQ 过于对齐，真实相机画面泛化不足。

缓解：

- 强化尺度、平移、旋转、背景和质量退化增强。
- 后续引入公开视频帧或自拍数据进行微调。

风险 2：头发、帽子边界精度不足。

缓解：

- 加 boundary loss。
- 使用更高输入尺寸如 384x384。
- 使用 teacher distillation。

风险 3：端上 delegate 掉回 CPU。

缓解：

- 模型只使用保守移动端算子。
- 导出后逐层检查 TFLite/Core ML 图。
- 在真机上做 early benchmark。

风险 4：多脸场景不稳定。

缓解：

- 首版明确只支持主脸/近景自拍。
- 如果产品必须多脸，后续应引入轻量 face detector 或其他 ROI 策略。

## 12. 第一阶段里程碑

M1：数据准备

- 合成 CelebAMask-HQ class-id mask。
- 生成 train/val/test split。
- 可视化检查 100 张样本。

M2：模型 baseline

- 实现 MobileNetV2 + LR-ASPP-lite。
- 训练 320x320 baseline。
- 输出 mIoU 和可视化结果。
- 使用 `scripts/evaluate.py` 生成 `metrics_val.json` 和样本预测网格图。

M3：蒸馏增强

- 准备 teacher checkpoint。
- 加入 soft-logit distillation。
- 对比 student baseline。

M4：端侧导出

- 导出 ONNX。
- 导出 TFLite FP16。
- 导出 Core ML FP16。
- 完成至少一台 Android 和一台 iOS 实测。
- ONNX 导出后使用 `scripts/export_onnx.py` 内置 graph check 检查是否存在移动端不友好的 op。

M5：优化迭代

- 根据端上 latency 选择 `256/320/384` 输入。
- 根据失败样本补增强或补数据。
- 固化 v1.0 模型配置。
- 按 `docs/benchmark_protocol.md` 记录 Android/iOS latency、内存、thermal 和失败样本。
