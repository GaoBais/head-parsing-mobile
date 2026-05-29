# Head Parsing Mobile

面向移动端实时运行的人头/人脸语义分割项目，目标部署平台为 Android 和 iOS。

本项目的目标模型不依赖 MediaPipe Face Landmarker。模型直接接收摄像头画面或缩放后的 RGB 图像，输出头部相关类别的逐像素语义分割结果。模型结构、训练流程和导出流程都以移动端友好算子为约束，优先兼容 Android LiteRT/TFLite GPU delegate 与 iOS Core ML。

## 目标

- 任务：实时人头/人脸语义分割
- 输入：RGB 图像，默认尺寸为 `320x320`
- 输出：逐像素类别 logits 或 mask
- 训练环境：Linux 服务器，使用 CUDA
- 部署环境：Android LiteRT/TFLite 与 iOS Core ML
- 主要数据集：CelebAMask-HQ
- 教师模型：现有 `../face-parsing` 项目中的 BiSeNet ResNet 模型
- 学生模型：MobileNetV2 或 MobileNetV3 编码器，加轻量 LR-ASPP/Fast-SCNN 风格解码器

## 文档

- [Agent 工作约定](AGENTS.md)
- [移动端人头分割方案](docs/mobile_head_parsing_plan.md)
- [项目目录结构](docs/project_structure.md)
- [实现进度记录](docs/progress.md)
- [设备 benchmark 协议](docs/benchmark_protocol.md)

## 目录概览

```text
head-parsing-mobile/
  configs/       训练、模型、导出和部署配置
  data/          原始数据链接、处理后标签、训练/验证/测试划分
  docs/          方案设计和实现文档
  scripts/       命令行入口脚本
  src/           Python 包源码
  tests/         单元测试和 smoke test
  tools/         数据检查、benchmark 和转换辅助工具
  benchmarks/    设备 benchmark 模板与运行记录
  weights/       本地 checkpoint 和导出模型
  outputs/       训练日志、评估报告和可视化样例
  third_party/   可选第三方代码快照或适配层
```

## 实现顺序

1. 构建 CelebAMask-HQ 标签合并、预处理与数据划分流程。
2. 实现模拟真实摄像头输入的移动端增强流程。
3. 实现 `HeadParsingMobile` 学生模型。
4. 训练教师模型，或复用现有 BiSeNet checkpoint 做蒸馏。
5. 训练并评估移动端学生模型。
6. 导出 ONNX、LiteRT/TFLite 和 Core ML 模型。
7. 在代表性 Android 和 iOS 设备上做延迟、稳定性和效果 benchmark。

## 本地开发说明

当前 Windows 环境只用于开发和 CPU 级别检查。本机使用宿主机 Python 3.13.7，不使用沙箱内 Python。完整 PyTorch/CUDA 训练、ONNX 导出验证、LiteRT/TFLite 转换和 Core ML 转换验证，后续在 Linux 服务器上执行。

本地 smoke test：

```bash
python -m unittest discover -s tests
```

## 训练

训练入口：

```bash
python scripts/train.py \
  --model-config configs/model_mobilev2_lraspp_320.yaml \
  --dataset-config configs/dataset_celebamask_hq.yaml \
  --train-config configs/train_student.yaml \
  --device cuda
```

如果服务器上还没有准备教师模型 checkpoint，可以先关闭 teacher 做训练流程 smoke test：

```bash
python scripts/train.py --device cuda --disable-teacher
```

## 评估

```bash
python scripts/evaluate.py \
  --checkpoint outputs/train/best.pt \
  --split val \
  --device cuda \
  --output-dir outputs/eval
```

## 模型导出

导出 ONNX：

```bash
python scripts/export_onnx.py \
  --checkpoint outputs/train/best.pt \
  --output weights/head_parsing_mobile_320.onnx \
  --device cpu
```

在 Linux 服务器上导出 LiteRT/TFLite：

```bash
pip install -r requirements-server.txt

python scripts/export_tflite.py \
  --checkpoint outputs/train/best.pt \
  --output weights/head_parsing_mobile_320_fp16.tflite \
  --precision fp16 \
  --device cpu
```

导出 Core ML：

```bash
python scripts/export_coreml.py \
  --checkpoint outputs/train/best.pt \
  --output weights/HeadParsingMobile320.mlpackage \
  --precision fp16 \
  --minimum-deployment-target ios15
```

## 设备测试前检查

在移动端 benchmark 前，先记录导出产物信息：

```bash
python tools/inspect_artifacts.py \
  weights/head_parsing_mobile_320_fp16.tflite \
  weights/HeadParsingMobile320.mlpackage \
  --output benchmarks/runs/latest/artifact_manifest.json
```

## 可视化工具

本地查看分割 mask：

```bash
python tools/visualize_masks.py \
  --image path/to/image.jpg \
  --mask path/to/mask.png \
  --output outputs/preview.jpg \
  --mode grid
```
