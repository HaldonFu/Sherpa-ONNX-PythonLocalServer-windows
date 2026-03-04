from huggingface_hub import snapshot_download
import os

print("🚀 正在从 Hugging Face 全速下载 FunASR-Nano-ONNX 究极版模型，文件较大，请耐心等待...")

# 设定模型保存路径
local_dir = os.path.join(os.getcwd(), "models", "FunASR-Nano-fp16-ONNX")

# 🌟 核心修改：使用 Hugging Face 的专用下载接口，并指向目标仓库
snapshot_download(
    repo_id='csukuangfj/sherpa-onnx-funasr-nano-fp16-2025-12-30', 
    local_dir=local_dir,
    # 可选：如果中途断网，再次运行会断点续传
    resume_download=True 
)

print(f"✅ 下载完成！模型已永久保存在: {local_dir}")