# Sherpa-ONNX Python Local Server

本项目是一个基于 `sherpa-onnx` 的语音识别（ASR）局域网服务端。提供了基于 Python 的服务，支持加载 FunASR-nano 和 Zipformer 等多种 ONNX 语音模型，可用于与 Unity 端或其他客户端进行非流式的语音交互。

## 核心文件说明

* `FunASR-nano-server.py` / `Zipformer-server.py`: 服务端主程序，负责加载对应模型并启动 WebSocket 监听。
* `download_model.py` / `download-zipformer-model.py`: 模型自动化下载脚本，用于从 Hugging Face 或 ModelScope 获取所需的 ONNX 模型文件。
* `hotwords.txt`: 热词配置文件，用于提升特定领域词汇的识别准确率。
* `system_prompt.txt`: 系统提示词配置文件。

## 🛠️ 环境配置与安装指南

推荐使用 Python 3.8 - 3.11 版本（我当前测试环境为 **Python 3.11.4**）。

### 1. 安装核心依赖

本项目依赖 `sherpa-onnx` 进行模型推理，以及 `websockets` 提供网络服务功能。

```bash
# 安装基础依赖
pip install websockets

# 安装 sherpa-onnx 
# 注意：如果你的电脑有 NVIDIA 显卡并希望使用 CUDA 加速，请安装 GPU 版本；否则安装 CPU 版本即可。
pip install sherpa-onnx
```

*注：GPU 版本可能需要自行编译 `.whl` 文件。自行编译后的安装命令如下：*
```bash
pip install XXX.whl
```

*如果在运行或打包时遇到缺少 `soundfile` 的报错，请额外安装：*
```bash
pip install soundfile
```

### 2. 下载模型文件

由于 GitHub 仓库限制，本项目不包含几十上百兆的 `.onnx` 模型文件及 `.dll` 运行库。

请根据需要，运行项目提供的下载脚本，它会自动将模型文件下载至本地的 `models/` 目录下：

```bash
# 下载 FunASR-nano 相关模型
python download_model.py

# 或者下载 Zipformer 相关模型
python download-zipformer-model.py
```

### 3. 运行服务端

环境配置完毕且模型下载完成后，即可启动服务端。以 FunASR-nano 为例：

```bash
python FunASR-nano-server.py
```

当控制台提示服务启动成功后，即可等待客户端（如 Unity）通过 WebSocket 进行连接。

最后这里的ultimate_build.py是sherpa-onnx的编译windows版本的python whi使用。跟sherpa-onnx中的setup.py一个意思。