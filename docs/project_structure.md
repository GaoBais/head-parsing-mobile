# 项目目录结构

```text
head-parsing-mobile/
  README.md
  configs/
    model_mobilev2_lraspp_320.yaml
    dataset_celebamask_hq.yaml
    train_student.yaml
    export_mobile.yaml
  data/
    raw/
      celebamask_hq/
    processed/
      celebamask_hq/
        images/
        masks/
    splits/
      train.txt
      val.txt
      test.txt
  docs/
    mobile_head_parsing_plan.md
    project_structure.md
  scripts/
    prepare_celebamask_hq.py
    train.py
    evaluate.py
    export_onnx.py
    export_tflite.py
    export_coreml.py
    benchmark_desktop.py
  src/
    datasets/
      celebamask_hq.py
      transforms.py
    models/
      head_parsing_mobile.py
      mobilenetv2.py
      lraspp.py
    training/
      losses.py
      distillation.py
      trainer.py
      metrics.py
    export/
      onnx.py
      tflite.py
      coreml.py
      graph_check.py
    deploy/
      preprocess.py
      postprocess.py
      palette.py
    utils/
      config.py
      seed.py
      visualization.py
      checkpoint.py
  tests/
    test_dataset.py
    test_model_shapes.py
    test_export_smoke.py
  tools/
    inspect_masks.py
    visualize_predictions.py
    profile_model.py
  notebooks/
  benchmarks/
    templates/
      device_latency.csv
      failure_case.json
      notes.md
  weights/
  outputs/
  third_party/
```

## 职责说明

`configs/`

保存所有可复现实验配置。训练、导出、数据路径和模型结构都通过配置文件声明，不把关键参数散落在脚本里。

`data/`

只放本地数据和 split 文件。大数据文件不提交 Git。`raw/` 保持原始数据结构，`processed/` 保存合成后的 image/mask 对。

`scripts/`

面向命令行使用的入口脚本。脚本只负责解析参数和调用 `src/` 中的模块。

`src/datasets/`

数据集读取、mask 合成后的 class-id 标签读取、训练增强和评估增强。

`src/models/`

移动端 student 模型实现。首版包含 MobileNetV2 encoder 和 LR-ASPP-lite decoder。

`src/training/`

loss、metric、trainer、蒸馏逻辑。teacher/student 的训练差异应放在这里。

`src/export/`

ONNX、TFLite/LiteRT、Core ML 导出和图检查逻辑。导出后需要检查是否包含不适合移动端的算子。

`src/deploy/`

与端侧一致的预处理和后处理逻辑，包括 resize/letterbox、mask restore、palette、可选时序平滑。

`tools/`

一次性检查工具和分析工具，比如 mask 可视化、类别分布统计、模型 profiling。

`benchmarks/`

端侧 benchmark 模板和实际运行记录。模板用于统一 Android/iOS 延迟、内存、失败样本和 artifact manifest 记录格式。

`tests/`

最小测试覆盖：数据 shape、模型输出 shape、导出 smoke test。

`weights/`

本地 checkpoint、teacher 权重和导出模型。默认不提交 Git。

`outputs/`

训练日志、评估报告、预测可视化和 benchmark 结果。默认不提交 Git。
