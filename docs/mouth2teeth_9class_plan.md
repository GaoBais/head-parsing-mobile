# Mouth2Teeth 9 类模型改造方案

本文档记录从现有 20 类 `mouth2teeth` 模型改造为 9 类移动端模型的方案。目标是在不破坏现有 20 类训练、评估、导出和 Android demo 路径的基础上，新增一套 9 类数据与训练路径，降低移动端输出 tensor 和 argmax 后处理成本。

## 背景

当前模型输出为：

```text
[1, 20, 320, 320]
```

移动端需要对 20 个 channel 做逐像素 argmax，输出 tensor 元素数为：

```text
20 * 320 * 320 = 2,048,000
```

9 类模型输出为：

```text
[1, 9, 320, 320]
```

输出 tensor 元素数为：

```text
9 * 320 * 320 = 921,600
```

输出和后处理规模降低约 `55%`。这不会改变模型主干的计算量，但可以明显降低端侧输出传输、内存访问和 argmax 后处理压力。

## 分支策略

在新分支中实施，保留 `main` 上已有 20 类逻辑：

```bash
git checkout -b feature/mouth2teeth-9class
```

原则：

- 不覆盖现有 20 类配置。
- 不修改已训练 20 类模型产物说明。
- 新增 9 类配置、数据目录、metadata 和文档。
- 训练、评估、导出命令显式使用 9 类配置。

## 目标类别

最终模型应保留 `background` 加 8 个业务类别，共 9 类。

| 新 ID | 新类别 | 原 20 类 ID | 中文 |
| ---: | --- | ---: | --- |
| 0 | `background` | 0 + 未保留类别 | 背景/不关心区域 |
| 1 | `skin` | 1 | 面部皮肤 |
| 2 | `l_brow` | 2 | 左眉毛 |
| 3 | `r_brow` | 3 | 右眉毛 |
| 4 | `mouth` | 11 | 口腔/嘴内部 |
| 5 | `u_lip` | 12 | 上嘴唇 |
| 6 | `l_lip` | 13 | 下嘴唇 |
| 7 | `teeth` | 19 | 牙齿 |
| 8 | `hair` | 17 | 头发 |

未保留旧类别全部合并到 `background`：

```text
l_eye, r_eye, eye_g, l_ear, r_ear, ear_r, nose, neck, neck_l, cloth, hat
```

## Mask 重映射

现有 20 类 mask 不能直接用于 9 类训练，因为 mask 像素值仍为 `0-19`。需要先生成新的 class-id mask，像素值范围为 `0-8`。

重映射表：

| 旧 ID | 旧类别 | 新 ID | 新类别 |
| ---: | --- | ---: | --- |
| 0 | `background` | 0 | `background` |
| 1 | `skin` | 1 | `skin` |
| 2 | `l_brow` | 2 | `l_brow` |
| 3 | `r_brow` | 3 | `r_brow` |
| 11 | `mouth` | 4 | `mouth` |
| 12 | `u_lip` | 5 | `u_lip` |
| 13 | `l_lip` | 6 | `l_lip` |
| 19 | `teeth` | 7 | `teeth` |
| 17 | `hair` | 8 | `hair` |
| 其他 | 未保留类别 | 0 | `background` |

建议新增工具：

```text
scripts/remap_masks.py
```

输入：

```text
data/processed/mouth2teeth/
  images/
  masks/
```

输出：

```text
data/processed/mouth2teeth_9cls/
  images/
  masks/
  metadata.json
```

split 建议单独放置：

```text
data/splits/mouth2teeth_9cls/
  train.txt
  val.txt
  test.txt
```

如果 9 类数据和 20 类数据来自同一批图片，可以复用 stem 列表，但建议单独保存 split 文件，便于后续审计和复现实验。

## 配置文件规划

新增配置，不覆盖现有文件：

```text
configs/model_mobilev2_lraspp_320_9cls.yaml
configs/dataset_mouth2teeth_9cls.yaml
configs/train_mouth2teeth_9cls.yaml
configs/mobile_metadata_mouth2teeth_9cls.json
```

关键配置：

```yaml
model:
  input_size: [320, 320]
  num_classes: 9
```

9 类 dataset config 应指向：

```yaml
dataset:
  name: mouth2teeth_9cls
  num_classes: 9
  image_dir: data/processed/mouth2teeth_9cls/images
  mask_dir: data/processed/mouth2teeth_9cls/masks
  splits:
    train: data/splits/mouth2teeth_9cls/train.txt
    val: data/splits/mouth2teeth_9cls/val.txt
    test: data/splits/mouth2teeth_9cls/test.txt
```

训练超参可以先继承 20 类 `train_mouth2teeth.yaml`：

```yaml
epochs: 80
batch_size: 32
precision: amp
optimizer: SGD
lr: 0.01
teacher.enabled: false
losses:
  ce_weight: 1.0
  dice_weight: 0.5
  boundary_weight: 0.2
  ohem: true
```

## 代码修改点

### 类别表和 palette

当前 `src/deploy/palette.py` 硬编码 20 类。9 类路径需要新增类别定义，同时保持 20 类可用。

建议新增：

```python
CLASS_NAMES_20
PALETTE_20
CLASS_NAMES_9
PALETTE_9
```

或者新增一个 class-set helper，让脚本从 dataset config / metadata 读取类别名和 palette。

目标是不让评估、可视化、TFLite smoke test 在 9 类模型上继续使用 20 类类别名。

### 水平翻转 label swap

当前 `src/datasets/transforms.py` 的 `LABEL_SWAPS` 按 20 类写死：

```python
2 <-> 3
4 <-> 5
7 <-> 8
```

9 类只应保留：

```python
2 <-> 3
```

否则会错误地把 9 类中的 `mouth/u_lip`、`teeth/hair` 互换。

建议让 `MobileTrainTransform` 支持传入 `label_swaps`，由 dataset config 或 train config 决定：

```yaml
augmentation:
  label_swaps:
    2: 3
    3: 2
```

20 类配置保持原 swap，9 类配置使用眉毛 swap。

### 评估和可视化

`scripts/evaluate.py` 当前从 `src.deploy.palette.CLASS_NAMES` 读取类别名。9 类训练后需要改成优先从 dataset config 或 metadata 读取：

```text
dataset.classes
```

否则 `metrics_val.json` 的 `per_class` 会错位。

`src/utils/visualization.py` 也需要支持传入 9 类 palette，不能默认总是 20 类 palette。

### TFLite smoke test

`tools/tflite_smoke_test.py` 当前默认使用 20 类 `CLASS_NAMES` 和 palette。9 类模型导出后输出 channel 为 9，需要让工具从：

```text
configs/mobile_metadata_mouth2teeth_9cls.json
```

读取：

```text
classes
palette
num_classes
```

### Android demo

Android demo 当前按 20 类写死：

```text
NUM_CLASSES = 20
TEETH_CLASS_ID = 19
```

9 类模型需要同步为：

```text
NUM_CLASSES = 9
TEETH_CLASS_ID = 7
```

建议不要覆盖当前 demo，后续可以新增：

```text
android-demo-9cls/
```

或者让 demo 从 metadata JSON 读取 class names 和 palette。

## 训练流程

### 生成 9 类数据

在服务器上执行 20 类到 9 类 mask remap，例如：

```bash
python scripts/remap_masks.py \
  --input-root data/processed/mouth2teeth \
  --output-root data/processed/mouth2teeth_9cls \
  --split-dir data/splits/mouth2teeth_9cls \
  --mapping mouth2teeth_9cls \
  --overwrite
```

具体参数以实际实现为准。

### 数据检查

必须检查：

- image 数量是否为 30000。
- mask 数量是否为 30000。
- image/mask stem 是否一一对应。
- mask 像素值是否只包含 `0-8`。
- `skin/l_brow/r_brow/mouth/u_lip/l_lip/teeth/hair` 在 train/val/test 中都有正样本。
- `teeth` 在 val/test 中必须有足够正样本，否则 IoU 没有评估意义。

### 小样本 smoke test

先使用少量样本做训练流程检查：

```bash
RUN_ID=train_mouth2teeth_9cls_smoke
mkdir -p outputs/$RUN_ID

PYTHONUNBUFFERED=1 python -u scripts/train.py \
  --model-config configs/model_mobilev2_lraspp_320_9cls.yaml \
  --dataset-config configs/dataset_mouth2teeth_9cls.yaml \
  --train-config configs/train_mouth2teeth_9cls.yaml \
  --output-dir outputs/$RUN_ID \
  --device cuda 2>&1 | tee outputs/$RUN_ID/train.log
```

### 完整训练

```bash
RUN_ID=train_mouth2teeth_9cls_full_v1
mkdir -p outputs/$RUN_ID

PYTHONUNBUFFERED=1 python -u scripts/train.py \
  --model-config configs/model_mobilev2_lraspp_320_9cls.yaml \
  --dataset-config configs/dataset_mouth2teeth_9cls.yaml \
  --train-config configs/train_mouth2teeth_9cls.yaml \
  --output-dir outputs/$RUN_ID \
  --device cuda 2>&1 | tee outputs/$RUN_ID/train.log
```

## 评估与导出

评估：

```bash
python scripts/evaluate.py \
  --checkpoint outputs/train_mouth2teeth_9cls_full_v1/best.pt \
  --model-config configs/model_mobilev2_lraspp_320_9cls.yaml \
  --dataset-config configs/dataset_mouth2teeth_9cls.yaml \
  --train-config configs/train_mouth2teeth_9cls.yaml \
  --split val \
  --device cuda \
  --output-dir outputs/eval_mouth2teeth_9cls_full_v1
```

TFLite 导出：

```bash
python scripts/export_tflite.py \
  --checkpoint outputs/train_mouth2teeth_9cls_full_v1/best.pt \
  --model-config configs/model_mobilev2_lraspp_320_9cls.yaml \
  --export-config configs/export_mobile.yaml \
  --output weights/head_parsing_mobile_320_mouth2teeth_9cls_fp16.tflite \
  --precision fp16 \
  --converter auto \
  --device cpu
```

TFLite smoke test：

```bash
python tools/tflite_smoke_test.py \
  --model weights/head_parsing_mobile_320_mouth2teeth_9cls_fp16.tflite \
  --metadata configs/mobile_metadata_mouth2teeth_9cls.json \
  --image data/processed/mouth2teeth_9cls/images/107.jpg \
  --output-dir outputs/tflite_smoke_mouth2teeth_9cls
```

## 风险和注意事项

- 9 类不是简单删输出 channel，必须重映射 mask。
- `background` 会吸收更多旧类别，背景类别语义会变宽；评估时要重点看前景类 IoU，不要只看 pixel accuracy。
- `nose/eye/ear/hat/cloth/neck` 被合并到 background 后，模型不会再区分这些区域。
- 水平翻转类别交换必须按 9 类重配，否则训练标签会被破坏。
- 9 类模型不能加载 20 类 checkpoint 的 decoder 权重；最多只复用 encoder 权重，当前配置可先从头训练。
- Android 端 `teeth` id 会从 `19` 变成 `7`，所有端侧 hardcode 必须同步。

## 验收标准

代码层：

- `python -m unittest discover -s tests` 通过。
- 20 类现有测试和配置不被破坏。
- 9 类 remap 工具能生成只含 `0-8` 的 masks。

数据层：

- 30000 张图片与 30000 张 mask 一一对应。
- train/val/test split 可复现。
- 9 类每类在 train/val/test 都有正样本统计。

训练层：

- 小样本 smoke test 能完成至少 1 个 epoch。
- 完整训练能保存 `last.pt`、`best.pt` 和 `train.log`。
- `metrics_val.json` 中 `per_class` 正确显示 9 类名称。

部署层：

- TFLite 输出 shape 为 `[1, 9, 320, 320]`。
- TFLite smoke test 能输出 `pred.png`、`color.png`、`overlay.jpg`。
- Android demo 或端侧集成使用新的 class id 和 palette。
