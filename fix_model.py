from modelscope import snapshot_download
import os
import shutil

# 1. 自动从国内镜像下载缺失的灵魂文件
print("正在从镜像站补全模型文件...")
temp_dir = snapshot_download('FunAudioLLM/Fun-ASR-MLT-Nano-2512')

# 2. 定位你那个报错的 models 文件夹路径
# 请确保这里的路径和你 server.py 里的 model_dir 一致
target_dir = os.path.join(os.getcwd(), "models", "Fun-ASR-MLT-Nano-2512")

# 3. 把缺失的 model.py 拷过去
src_file = os.path.join(temp_dir, "model.py")
if os.path.exists(src_file):
    shutil.copy(src_file, target_dir)
    print(f"✅ 成功！model.py 已补全到: {target_dir}")
else:
    print("❌ 镜像站也没找到 model.py，请检查模型名称是否输入正确。")