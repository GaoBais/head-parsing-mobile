# Mouth2Teeth 9 类回归测试结果

日期：2026-06-12
分支：`feature/mouth2teeth-9class`

## 测试目标

验证 9 类改造在不破坏 20 类路径的前提下可用：

- 20 类默认类别表保持兼容。
- 新增 9 类类别表、palette 和 metadata 正确。
- 20 类 mask 可以稳定重映射为 9 类 class-id mask。
- 9 类训练增强只交换 `l_brow/r_brow`，不会错误交换 `mouth/u_lip` 或 `teeth/hair`。
- 评估报告使用 dataset config 中的 9 类类别名。
- TFLite smoke test 可以从 metadata 读取 9 类类别和 palette。
- 训练、评估、remap 工具命令入口可用。

## 测试用例

| 测试文件 | 覆盖内容 |
| --- | --- |
| `tests/test_palette_9cls.py` | 9 类顺序、9 类 palette、class set 查询 |
| `tests/test_remap_masks.py` | 20 类到 9 类映射表、remap 后 mask 值范围、metadata 统计和 split 复制 |
| `tests/test_transforms.py` | 9 类 label swap 只交换左右眉毛，保留 mouth/lip/teeth/hair ID |
| `tests/test_train_config.py` | `train.augmentation.label_swaps` 可从配置传入 transform |
| `tests/test_visualization.py` | 9 类 palette 和 class histogram 可正确工作 |
| `tests/test_tflite_smoke.py` | 9 类 logits 解码、9 类 metadata 读取 |
| `tests/test_evaluate_config.py` | 评估阶段从 dataset config 解析 9 类类别名，metrics 按配置写入 |

## 执行结果

### 单元测试

命令：

```powershell
python -m unittest discover -s tests
```

结果：

```text
Ran 34 tests in 0.217s
OK (skipped=8)
```

说明：8 个 skipped 是当前 Windows 本地环境缺少 PyTorch/CoreML 等运行依赖时的既有跳过项，不属于 9 类改造失败。

### 静态编译检查

命令：

```powershell
python -m py_compile scripts\remap_masks.py scripts\train.py scripts\evaluate.py tools\tflite_smoke_test.py src\deploy\palette.py src\datasets\transforms.py src\utils\visualization.py tests\test_evaluate_config.py tests\test_remap_masks.py tests\test_palette_9cls.py tests\test_transforms.py tests\test_tflite_smoke.py
```

结果：通过，无输出。

### 9 类配置加载检查

命令：

```powershell
@'
from src.utils.config import load_yaml
for path in [
    'configs/model_mobilev2_lraspp_320_9cls.yaml',
    'configs/dataset_mouth2teeth_9cls.yaml',
    'configs/train_mouth2teeth_9cls.yaml',
]:
    cfg = load_yaml(path)
    print(path, 'ok')
print('model num_classes', load_yaml('configs/model_mobilev2_lraspp_320_9cls.yaml')['model']['num_classes'])
print('dataset num_classes', load_yaml('configs/dataset_mouth2teeth_9cls.yaml')['dataset']['num_classes'])
print('train label_swaps', load_yaml('configs/train_mouth2teeth_9cls.yaml')['train']['augmentation']['label_swaps'])
'@ | python -
```

结果：

```text
configs/model_mobilev2_lraspp_320_9cls.yaml ok
configs/dataset_mouth2teeth_9cls.yaml ok
configs/train_mouth2teeth_9cls.yaml ok
model num_classes 9
dataset num_classes 9
train label_swaps {2: 3, 3: 2}
```

### Metadata JSON 检查

命令：

```powershell
python -c "import json; json.load(open('configs/mobile_metadata_mouth2teeth_9cls.json', encoding='utf-8')); print('9cls metadata json ok')"
```

结果：

```text
9cls metadata json ok
```

### CLI 入口检查

命令：

```powershell
python scripts\remap_masks.py --help
python scripts\train.py --help
python scripts\evaluate.py --help
python tools\tflite_smoke_test.py --help
```

结果：全部返回 usage 信息，退出码为 0。

## 当前未覆盖项

以下检查需要服务器或完整数据集：

- 30000 张完整数据的 9 类 remap 实际执行。
- 9 类 train/val/test 每类正样本统计，尤其是 `teeth`。
- CUDA 小样本训练 smoke test。
- 9 类完整训练。
- 9 类 TFLite 导出和真实 TFLite runtime 推理。
- Android 端 9 类模型集成和耗时对比。

## 下一步命令

生成 9 类 processed 数据：

```bash
python scripts/remap_masks.py \
  --input-root data/processed/mouth2teeth \
  --output-root data/processed/mouth2teeth_9cls \
  --split-dir data/splits/mouth2teeth_9cls \
  --source-split-dir data/splits/mouth2teeth \
  --mapping mouth2teeth_9cls \
  --overwrite
```

完整训练：

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
