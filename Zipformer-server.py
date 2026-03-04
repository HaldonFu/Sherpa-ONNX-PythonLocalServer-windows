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
# 1. 动态获取模型路径
# ==========================================
if getattr(sys, 'frozen', False):
    application_path = os.path.dirname(sys.executable)
else:
    application_path = os.path.dirname(os.path.abspath(__file__))

model_dir = os.path.join(application_path, "models", "sherpa-onnx-zipformer-zh-en-2023-11-22")

# ==========================================
# 2. Zipformer Transducer (纯CPU版) + 全自动 BPE 热词引擎
# ==========================================
import sentencepiece as spm

print("="*40)
print(f"正在加载 Zipformer Transducer (纯CPU版) + 中英双语热词引擎...")

hotwords_path = os.path.join(application_path, "hotwords.txt")
hotwords_bpe_path = os.path.join(application_path, "hotwords_bpe.txt")

# 自动生成模板
if not os.path.exists(hotwords_path):
    with open(hotwords_path, "w", encoding="utf-8") as f:
        f.write("ulcers :4.5\ncaries :4.0\n龋坏 :2.5\n嵌塞 :2.0\n干槽症 :3.0\n根管治疗 :3.0")

# 用 SentencePiece 自动把人类词汇翻译成 AI BPE 碎片
try:
    sp = spm.SentencePieceProcessor()
    # 💡 确保你的模型文件夹里有 bpe.model，如果没有就改成 bbpe.model
    sp.load(os.path.join(model_dir, "bbpe.model")) 
    
    bpe_lines = []
    with open(hotwords_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            
            parts = line.split(":")
            
            # 致命级修复：必须用小写 lower()！底层 tokens.txt 里的英文全都是小写碎片
            word = parts[0].strip().lower() 
            score = parts[1].strip() if len(parts) > 1 else "2.0"
            
            # 将单词切碎成带有特殊空格的 BPE Token 碎片
            pieces = sp.encode_as_pieces(word)
            pieces_str = " ".join(pieces)
            bpe_lines.append(f"{pieces_str} :{score}")
    
    # 保存一份底层的“碎片版热词”给 C++ 引擎读取
    with open(hotwords_bpe_path, "w", encoding="utf-8") as f:
        f.write("\n".join(bpe_lines))
        
except Exception as e:
    print(f"分词引擎启动失败，详细报错: {e}")
    sys.exit(1)

try:
    # 🌟 核心修复：用回你原先正确的 from_transducer 方法！
    recognizer = sherpa_onnx.OfflineRecognizer.from_transducer(
        encoder=os.path.join(model_dir, "encoder-epoch-34-avg-19.onnx"),
        decoder=os.path.join(model_dir, "decoder-epoch-34-avg-19.onnx"),
        joiner=os.path.join(model_dir, "joiner-epoch-34-avg-19.onnx"),
        tokens=os.path.join(model_dir, "tokens.txt"),
        
        provider="cpu",
        num_threads=4,
        decoding_method="modified_beam_search", 
        
        # 传递处理好的 BPE 碎片文件
        hotwords_file=hotwords_bpe_path, 
        hotwords_score=2.0 
    )
    print("Zipformer 引擎加载成功！CPU 准备就绪，1660 显卡已彻底解放！")
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
infer_lock = threading.Lock() # 虽然跑 CPU，但依然留着排队锁防高并发踩踏

def process_audio(raw_bytes: bytes) -> str:
    if not raw_bytes: return ""
    
    samples_np = np.frombuffer(raw_bytes, dtype=np.float32)
    if len(samples_np) < 1600: return ""

    with infer_lock:
        stream = recognizer.create_stream()
        stream.accept_waveform(16000, samples_np)
        recognizer.decode_stream(stream)
        result_text = stream.result.text.strip()
    
    # 净化逻辑
    result_text = result_text.replace("/sil", "").strip()
    if result_text in ["", "嗯", "啊", "呃", "哦", "嗯嗯"]:
        return ""
        
    return result_text

# ==========================================
# 5. API 接口
# ==========================================
@app.post("/recognize")
async def recognize_api(request: Request):
    raw_bytes = await request.body()
    result_text = await asyncio.to_thread(process_audio, raw_bytes)
    if result_text:
        print(f"[识别结果] -> {result_text}")
    return {"text": result_text}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="warning")