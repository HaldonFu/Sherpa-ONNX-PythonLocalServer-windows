import uvicorn
from fastapi import FastAPI, Request
import sherpa_onnx
import numpy as np
import os
import sys
import socket
import threading
import asyncio

app = FastAPI()

# ==========================================
# 1.动态获取模型路径
# ==========================================
if getattr(sys, 'frozen', False):
    application_path = os.path.dirname(sys.executable)
else:
    application_path = os.path.dirname(os.path.abspath(__file__))

model_dir = os.path.join(application_path, "models", "FunASR-Nano-int8-ONNX")

# ==========================================
# 2. FunASR-Nano INT8 (纯CPU版) + 外置配置热词引擎
# ==========================================
print("="*40)
print(f"正在加载 FunASR-Nano INT8 (纯CPU极速版) + 动态热词引擎...")

# --- 读取外部配置文件 ---
hotwords_path = os.path.join(application_path, "hotwords.txt")
prompt_path = os.path.join(application_path, "system_prompt.txt")

# 如果没有配置文件，自动生成默认的
if not os.path.exists(hotwords_path):
    with open(hotwords_path, "w", encoding="utf-8") as f:
        f.write("ulcers\ncaries\n龋坏\n嵌塞\n干槽症\n根管治疗") 
if not os.path.exists(prompt_path):
    with open(prompt_path, "w", encoding="utf-8") as f:
        f.write("You are a patient in a dental clinic. 这是一个口腔齿科问诊环境。")

# 加载配置内容 (将热词的换行符替换为空格，以符合引擎格式)
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
        num_threads=8,   
        
        decoding_method="greedy_search",
        hotwords=custom_hotwords,    
        system_prompt=custom_prompt, 
        itn=True
    )
    print("✅ ASR模型加载成功！CPU 准备就绪，1660显卡彻底放假！")
except Exception as e:
    print(f"❌ 模型加载失败，详细报错: {e}")
    sys.exit(1)
print("="*40)

# ==========================================
# 3. UDP 雷达监听
# ==========================================
def udp_listener():
    udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    udp_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    udp_socket.bind(("0.0.0.0", 8888))
    print("📡 局域网自动嗅探雷达已开启 (UDP: 8888)")
    while True:
        try:
            data, addr = udp_socket.recvfrom(1024)
            if data.decode('utf-8') == "WhoIsSenseVoiceServer":
                udp_socket.sendto("SenseVoiceServer_Here".encode('utf-8'), addr)
        except: pass

threading.Thread(target=udp_listener, daemon=True).start()

# ==========================================
# 4. 核心解码逻辑 (加锁防踩踏版)
# ==========================================
# 🌟 顺手把变量名改成了更贴切的 infer_lock，反正现在不用 GPU 了
infer_lock = threading.Lock()

def process_audio(raw_bytes: bytes) -> str:
    if not raw_bytes: return ""
    
    samples_np = np.frombuffer(raw_bytes, dtype=np.float32)
    if len(samples_np) < 1600: return ""

    # 强行让所有线程排队进 CPU！
    with infer_lock:
        stream = recognizer.create_stream()
        stream.accept_waveform(16000, samples_np)
        recognizer.decode_stream(stream)
        result_text = stream.result.text.strip()
    
    return result_text

# ==========================================
# 5. API 接口
# ==========================================
@app.post("/recognize")
async def recognize_api(request: Request):
    raw_bytes = await request.body()
    result_text = await asyncio.to_thread(process_audio, raw_bytes)
    if result_text:
        print(f"🧠 [识别结果] -> {result_text}")
    return {"text": result_text}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="warning")