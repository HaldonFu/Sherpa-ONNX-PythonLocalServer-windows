import os
import urllib.request
import tarfile
import sys

# ==========================================
# ⚙️ 配置区
# ==========================================
MODEL_NAME = "sherpa-onnx-zipformer-zh-en-2023-11-22"
FILE_NAME = f"{MODEL_NAME}.tar.bz2"

# 默认官方 GitHub 下载源
GITHUB_URL = f"https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/{FILE_NAME}"

# 💡 如果你发现下载报错或卡住不动，请注释掉上面那行，把下面这行前面的 # 删掉，启用国内加速镜像！
DOWNLOAD_URL = GITHUB_URL
# DOWNLOAD_URL = f"https://mirror.ghproxy.com/{GITHUB_URL}"

# 自动定位到 models 文件夹
application_path = os.path.dirname(os.path.abspath(__file__))
models_dir = os.path.join(application_path, "models")
os.makedirs(models_dir, exist_ok=True) # 如果没有 models 文件夹会自动建一个
file_path = os.path.join(models_dir, FILE_NAME)

# ==========================================
# 🚀 核心逻辑
# ==========================================
def download_progress(block_num, block_size, total_size):
    """极其丝滑的进度条引擎"""
    downloaded = block_num * block_size
    if total_size > 0:
        percent = min(100.0, downloaded * 100 / total_size)
        sys.stdout.write(f"\r⏳ 下载进度: {percent:.1f}% ({downloaded/(1024*1024):.1f}MB / {total_size/(1024*1024):.1f}MB)")
        sys.stdout.flush()

print("="*50)
print(f"🚀 开始获取官方双语神装：{MODEL_NAME}")
print("="*50)

try:
    # 1. 下载阶段
    if not os.path.exists(file_path):
        print(f"🔗 正在连接服务器...\n请求地址: {DOWNLOAD_URL}")
        urllib.request.urlretrieve(DOWNLOAD_URL, file_path, download_progress)
        print("\n✅ 下载彻底完成！")
    else:
        print(f"\n📦 压缩包已存在，直接跳过下载阶段。")

    # 2. 解压阶段
    print(f"🛠️ 正在执行高能解压，大概需要十几秒，请稍候...")
    with tarfile.open(file_path, "r:bz2") as tar:
        tar.extractall(path=models_dir)
    print(f"🎉 部署成功！你的新模型已安家在: {os.path.join(models_dir, MODEL_NAME)}")
    
    # 3. 扫尾阶段：删掉占地方的压缩包
    os.remove(file_path)
    print("🧹 原始压缩包已自动清理。")
    
except Exception as e:
    print(f"\n❌ 行动失败，详细死因: {e}")
    print("💊 抢救建议：如果一直报 Timeout 超时，请去代码里把 DOWNLOAD_URL 换成下面那个 ghproxy 加速链接再试一次！")
print("="*50)