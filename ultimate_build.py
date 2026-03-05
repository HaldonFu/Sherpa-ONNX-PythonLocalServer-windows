import os
import shutil
import subprocess
from pathlib import Path

print("🗑️ 第 0 步：核弹清理！删除被污染的 CPU 缓存...")
shutil.rmtree("build", ignore_errors=True)
shutil.rmtree("dist", ignore_errors=True)

print("🔌 第 0.5 步：强行在代码级注入显卡灵魂！...")
# 🌟 核心杀招：彻底杜绝终端环境变量不生效的问题！
os.environ["SHERPA_ONNX_CMAKE_ARGS"] = "-DSHERPA_ONNX_ENABLE_GPU=ON"
os.environ["SHERPA_ONNX_ENABLE_GPU"] = "ON"

print("🔍 第 1 步：修复官方瞎眼的打包配置...")
setup_file = "setup.py"
with open(setup_file, "r", encoding="utf-8") as f:
    content = f.read()

# 强制把 lib 目录和二进制文件加入打包白名单
if '"sherpa_onnx.lib"' not in content:
    content = content.replace('packages=["sherpa_onnx"],', 
                              'packages=["sherpa_onnx", "sherpa_onnx.lib"],\n    package_data={"sherpa_onnx.lib": ["*.dll", "*.pyd", "*.so"]},')
    with open(setup_file, "w", encoding="utf-8") as f:
        f.write(content)

# 提前建好官方需要的空文件夹防撞墙
Path("build/sherpa_onnx/bin").mkdir(parents=True, exist_ok=True)

print("🔥 第 2 步：启动纯血 C++ GPU 引擎编译 (需要几分钟下载和编译)...")
subprocess.run(["python", "setup.py", "build_ext"], check=True)

print("📦 第 3 步：人工捕获散落的底层库...")
lib_dir = Path("sherpa-onnx/python/sherpa_onnx/lib")
lib_dir.mkdir(parents=True, exist_ok=True)
with open(lib_dir / "__init__.py", "w") as f:
    f.write("") 

found_pyd = False
for p in Path("build").rglob("_sherpa_onnx*.pyd"):
    shutil.copy(p, lib_dir)
    print(f"  [+] 成功捕获核心脑干: {p.name}")
    found_pyd = True

found_cuda = False
for p in Path("build").rglob("*.dll"):
    if "onnxruntime" in p.name or "sherpa-onnx" in p.name:
        shutil.copy(p, lib_dir)
        print(f"  [+] 成功捕获显卡算子: {p.name}")
        if "cuda" in p.name:
            found_cuda = True

if not found_pyd:
    print("❌ 警告：没有找到 .pyd 文件，编译可能因网络问题失败了！")
    exit(1)

if not found_cuda:
    print("⚠️ 极其危险：你的包里没抓取到带有 cuda 字样的 dll！显卡没成功注入！")

print("🚀 第 4 步：强行封装巨无霸...")
subprocess.run(["python", "setup.py", "bdist_wheel"], check=True)
print("🎉 彻底通关！快去 dist 文件夹提取你的满血 GPU 包吧！")