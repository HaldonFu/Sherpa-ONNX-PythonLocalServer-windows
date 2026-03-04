import os
# ==========================================
# 0. 🚀 核心防爆破配置：限制底层数学库的线程漫游
# 必须放在最上面，甚至在 import uvicorn 和 fastapi 之前！
# ==========================================
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import uvicorn
from fastapi import FastAPI, Request
import sherpa_onnx
import numpy as np
import sys
import socket
import threading
import asyncio
from concurrent.futures import ThreadPoolExecutor
import time
from datetime import datetime

app = FastAPI()

# ==========================================
# 0.5 🚀 i7-12700KF 满血并发配置区
# ==========================================
# 你的 CPU 有 20 个逻辑线程。这里设为 10 并发，
# 配合下面模型底层的 2 线程，刚好 10 * 2 = 20，榨干算力不排队！
MAX_CONCURRENT_RECOGNIZERS = 10 

# 创建全局线程池
executor = ThreadPoolExecutor(max_workers=MAX_CONCURRENT_RECOGNIZERS)

# ==========================================
# 1. 动态获取模型路径
# ==========================================
if getattr(sys, 'frozen', False):
    application_path = os.path.dirname(sys.executable)
else:
    application_path = os.path.dirname(os.path.abspath(__file__))

model_dir = os.path.join(application_path, "models", "FunASR-Nano-ONNX")

# ==========================================
# 2. 加载 FunASR-Nano + 外置配置热词引擎
# ==========================================
print("="*40)
print(f"正在加载 FunASR-Nano (ONNX极速版) + 动态热词引擎...")

hotwords_path = os.path.join(application_path, "hotwords.txt")
prompt_path = os.path.join(application_path, "system_prompt.txt")

if not os.path.exists(hotwords_path):
    with open(hotwords_path, "w", encoding="utf-8") as f:
        f.write("ulcers\ncaries\n龋坏\n嵌塞\n干槽症\n根管治疗")
if not os.path.exists(prompt_path):
    with open(prompt_path, "w", encoding="utf-8") as f:
        f.write("You are a patient in a dental clinic. 这是一个口腔齿科问诊环境。")

with open(hotwords_path, "r", encoding="utf-8") as f:
    custom_hotwords = f.read().replace('\n', ' ').strip()
with open(prompt_path, "r", encoding="utf-8") as f:
    custom_prompt = f.read().strip()

print(f"当前加载热词: {custom_hotwords[:30]}...")
print(f"当前系统提示词: {custom_prompt[:30]}...")

try:
    recognizer = sherpa_onnx.OfflineRecognizer.from_funasr_nano(
        encoder_adaptor=os.path.join(model_dir, "encoder_adaptor.int8.onnx"),
        llm=os.path.join(model_dir, "llm.int8.onnx"),  
        embedding=os.path.join(model_dir, "embedding.int8.onnx"),
        tokenizer=os.path.join(model_dir, "Qwen3-0.6B").replace("\\", "/"),
        provider="cpu",
        # 🚀 保持 2 个线程用于单个任务解码
        num_threads=2, 
        decoding_method="greedy_search",
        hotwords=custom_hotwords,  
        system_prompt=custom_prompt, 
        itn=True
    )
    print(f"Nano 模型加载成功! (底层线程数: 2 | 全局并发上限: {MAX_CONCURRENT_RECOGNIZERS})")
except Exception as e:
    print(f"模型加载失败，详细报错: {e}")
    sys.exit(1)
print("="*40)

# ==========================================
# 3. UDP 雷达监听
# ==========================================
def udp_listener():
    udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    udp_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    udp_socket.bind(("0.0.0.0", 8888))
    print("局域网自动嗅探雷达已开启 (UDP: 8888)")
    while True:
        try:
            data, addr = udp_socket.recvfrom(1024)
            if data.decode('utf-8') == "WhoIsSenseVoiceServer":
                udp_socket.sendto("SenseVoiceServer_Here".encode('utf-8'), addr)
        except: pass

threading.Thread(target=udp_listener, daemon=True).start()

# ==========================================
# 4. 核心解码逻辑
# ==========================================
def process_audio(raw_bytes: bytes) -> str:
    if not raw_bytes: return ""
    
    samples_np = np.frombuffer(raw_bytes, dtype=np.float32)
    if len(samples_np) < 1600: return ""

    stream = recognizer.create_stream()
    stream.accept_waveform(16000, samples_np)
    recognizer.decode_stream(stream)
    
    return stream.result.text.strip()

# ==========================================
# 5. API 接口 (带详细日志 + 音频时长版)
# ==========================================
@app.post("/recognize")
async def recognize_api(request: Request):
    # 1. 记录接收时间
    start_time = time.time()
    receive_time_str = datetime.fromtimestamp(start_time).strftime('%H:%M:%S.%f')[:-3]
    
    # 2. 获取客户端 IP
    client_ip = request.client.host if request.client else "未知IP"
    
    # 3. 接收字节数据
    raw_bytes = await request.body()
    
    # 🚀 4. 瞬间计算音频真实时长 (0 耗时)
    # 字节数 / 4 (float32大小) / 16000 (采样率) = 音频秒数
    audio_duration = len(raw_bytes) / 4 / 16000.0
    
    # 5. 丢给底层去解码
    loop = asyncio.get_running_loop()
    result_text = await loop.run_in_executor(executor, process_audio, raw_bytes)
    
    # 6. 计算服务端纯处理耗时
    cost_time = time.time() - start_time
    
    # 7. 打印超详细流水日志
    if result_text:
        print(f"[{receive_time_str}] 来源: {client_ip:<15} | 音长: {audio_duration:>4.2f}s | 耗时: {cost_time:>5.2f}s | 结果: {result_text}")
    else:
        print(f"[{receive_time_str}] 来源: {client_ip:<15} | 音长: {audio_duration:>4.2f}s | 耗时: {cost_time:>5.2f}s | 结果: [未识别出有效内容/静音]")
        
    return {"text": result_text}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="warning")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="warning")