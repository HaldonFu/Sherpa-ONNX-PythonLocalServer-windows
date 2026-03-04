import uvicorn
from fastapi import FastAPI, Request
import sherpa_onnx
import numpy as np
import os
import sys
import socket
import threading

app = FastAPI()

if getattr(sys, 'frozen', False):
    application_path = os.path.dirname(sys.executable)
else:
    application_path = os.path.dirname(os.path.abspath(__file__))

model_dir = os.path.join(application_path, "models", "SenseVoiceSmall")
model_path = os.path.join(model_dir, "model.int8.onnx")
tokens_path = os.path.join(model_dir, "tokens.txt")
#hotwords_path = os.path.join(application_path, "hotwords.txt")

print("="*40)
print(f"正在加载模型...")
try:
    recognizer = sherpa_onnx.OfflineRecognizer.from_sense_voice(
        model=model_path,
        tokens=tokens_path,
        num_threads=4,
        use_itn=True
    )
    print("SenseVoice 模型加载成功！")
except Exception as e:
    print(f"模型加载失败，详细报错: {e}")
    sys.exit(1)
print("="*40)

# ==========================================
# 🌟 新增功能：后台 UDP 广播监听
# ==========================================
def udp_listener():
    # 创建 UDP 套接字，监听 8888 端口
    udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    udp_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    udp_socket.bind(("0.0.0.0", 8888))
    print("局域网自动嗅探雷达已开启 (UDP 端口: 8888)")
    
    while True:
        try:
            data, addr = udp_socket.recvfrom(1024)
            message = data.decode('utf-8')
            if message == "WhoIsSenseVoiceServer":
                print(f"收到来自 {addr[0]} 的寻呼，正在回复我的位置...")
                # 收到暗号，立刻回复确认信息
                udp_socket.sendto("SenseVoiceServer_Here".encode('utf-8'), addr)
        except Exception as e:
            pass

# 启动 UDP 监听子线程 (daemon=True 保证主程序退出时它也跟着退出)
threading.Thread(target=udp_listener, daemon=True).start()

# ==========================================
# 原有的 HTTP 识别接口
# ==========================================
@app.post("/recognize")
async def recognize(request: Request):
    raw_bytes = await request.body()
    if not raw_bytes: return {"text": ""}
    samples = np.frombuffer(raw_bytes, dtype=np.float32)
    stream = recognizer.create_stream()
    stream.accept_waveform(16000, samples)
    recognizer.decode_stream(stream)
    print(f"[识别结果] -> {stream.result.text}")
    return {"text": stream.result.text}

if __name__ == "__main__":
    print("局域网语音识别服务器已启动！等待 Unity 客户端连接...")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="warning")
