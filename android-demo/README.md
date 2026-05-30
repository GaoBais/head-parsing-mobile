# Head Parsing Android Demo

这是一个最小 Android demo，用于验证 `head_parsing_mobile_320_teeth_fp16.tflite` 在 Android 端的输入、输出和后处理逻辑。它不包含相机采集，只从 assets 读取一张测试图，运行模型并显示彩色分割 mask。

## 需要安装

- Android Studio，或 Android command-line tools
- Android SDK Platform 36.1
- Android SDK Build-Tools
- Android SDK Platform-Tools
- JDK 17 或更高版本

当前本机已检测到 Java 21、Android SDK Platform 36.1、Build-Tools 和 Platform-Tools。`adb` 可通过 SDK 目录使用，但没有加入 PATH；本机仍没有全局 Gradle 命令。推荐用 Android Studio 打开本目录完成 Gradle Sync 和运行。

## 准备模型和样本图

在仓库根目录执行：

```powershell
Copy-Item weights\head_parsing_mobile_320_teeth_fp16.tflite android-demo\app\src\main\assets\
Copy-Item data\processed\mouth2teeth\images\107.jpg android-demo\app\src\main\assets\sample_107.jpg
```

assets 内的模型和图片是运行产物，不提交到 git。

## 运行

推荐先用 Android Studio 打开 `android-demo` 目录，等待 Gradle Sync 完成，然后点击 Run。

命令行构建需要本机已经安装 Gradle：

```powershell
cd android-demo
gradle :app:assembleDebug
```

如果使用 Android Studio，它通常会根据项目配置下载 Android Gradle Plugin 依赖；如果只使用命令行工具，需要手动准备 Gradle。

## Gradle 下载失败处理

如果 Android Studio 报错：

```text
Could not install Gradle distribution from 'https://services.gradle.org/distributions/gradle-9.0.0-bin.zip'
java.nio.file.NoSuchFileException: ...gradle-9.0.0-bin.zip
```

通常是 Gradle distribution 缓存里只留下了 `.part` 或 `.lck` 文件。关闭 Android Studio 后删除损坏缓存，再重新打开项目并执行 Gradle Sync：

```powershell
Remove-Item "$env:USERPROFILE\.gradle\wrapper\dists\gradle-9.0.0-bin" -Recurse -Force
```

如果仍然失败，优先检查网络是否能访问 `https://services.gradle.org/` 和 GitHub，因为 Gradle distribution 会重定向到 GitHub 下载。

## 验证点

首次运行后检查页面输出：

- input shape 是否为 `[1, 3, 320, 320]`
- output shape 是否为 `[1, 20, 320, 320]`
- `teeth pixels` 是否符合本地 smoke test 的预期
- 彩色 mask 是否与本地 `outputs/tflite_smoke_local_107` 视觉接近

本地对照命令：

```powershell
.\.venv-tflite\Scripts\python.exe tools\tflite_smoke_test.py `
  --model weights\head_parsing_mobile_320_teeth_fp16.tflite `
  --image data\processed\mouth2teeth\images\107.jpg `
  --output-dir outputs\tflite_smoke_local_107 `
  --num-threads 4
```
