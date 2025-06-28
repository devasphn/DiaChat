import os
import sys
import argparse
import asyncio
import queue
import threading
import random

import numpy as np
import resampy
import webrtcvad
import ollama
import torch

from pywhispercpp.model import Model
from dia.model import Dia
from transformers import AutoConfig
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
import uvicorn

# ─── CONFIG ───────────────────────────────────────────────────────────────────
MODELS_DIR = os.getenv("MODELS_DIR", "./models")
os.environ["HF_HUB_CACHE"] = MODELS_DIR

WHISPER_MODEL_NAME = "base.en"
DIA_MODEL_NAME     = "nari-labs/Dia-1.6B"
OLLAMA_HOST        = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL       = os.getenv("OLLAMA_MODEL", "llama3.2:latest")

TTS_SEED           = 42
FIXED_VOICE_PROMPT = "[S1]"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[main] Device: {DEVICE}")
print(f"[main] PyTorch: {torch.__version__}, NumPy: {np.__version__}")

def set_deterministic_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark     = False
    print(f"[voice] Seed set to {seed}")

set_deterministic_seed(TTS_SEED)

# ─── PRE-FLIGHT CHECKS ─────────────────────────────────────────────────────────
def check_audio():
    print("[check] Audio processing...")
    vad = webrtcvad.Vad(2)
    vad.is_speech(np.zeros(480, dtype=np.int16).tobytes(), 16000)
    resampy.resample(np.random.random(4096).astype(np.float32), 48000, 16000)
    print("[check] ✅ Audio OK")
    return True

def check_whisper():
    print("[check] Whisper model...")
    Model(WHISPER_MODEL_NAME)
    print("[check] ✅ Whisper OK")
    return True

def check_dia():
    print("[check] Dia TTS model...")
    # reload config with both encoder & decoder present
    config = AutoConfig.from_pretrained(DIA_MODEL_NAME)
    set_deterministic_seed(TTS_SEED)
    dia = Dia.from_pretrained(
        DIA_MODEL_NAME,
        config=config,
        device=DEVICE,
        compute_dtype="float16"
    )
    print("[check] ✅ Dia loaded")
    set_deterministic_seed(TTS_SEED)
    audio = dia.generate(
        f"{FIXED_VOICE_PROMPT} Hello test",
        use_torch_compile=False,
        verbose=False
    )
    print(f"[check] ✅ Dia generated {audio.shape[0]} samples")
    return True

def check_ollama():
    print("[check] Ollama service...")
    client = ollama.Client(host=OLLAMA_HOST)
    models = [m.model if hasattr(m, "model") else m["name"]
              for m in client.list().get("models", [])]
    print(f"[check] Available: {models}")
    if OLLAMA_MODEL not in models:
        client.pull(OLLAMA_MODEL)
    client.chat(model=OLLAMA_MODEL, messages=[{"role":"user","content":"ping"}])
    print("[check] ✅ Ollama OK")
    return True

def run_preflight_checks():
    print("\n🚀 Pre-flight checks:\n")
    for name, fn in [
        ("Audio"   , check_audio),
        ("Whisper" , check_whisper),
        ("Dia TTS" , check_dia),
        ("Ollama"  , check_ollama),
    ]:
        print(f"--- {name} ---")
        if not fn():
            print(f"❌ {name} failed. Exiting.")
            sys.exit(1)
    print("\n🎉 All checks passed!\n")

# ─── FASTAPI + WEBSOCKET ───────────────────────────────────────────────────────
app      = FastAPI()
clients  = set()
to_llm   = queue.Queue()
text_q   = queue.Queue()
to_tts   = queue.Queue()
ws_queue = queue.Queue()

@app.websocket("/ws/audio")
async def ws_audio(ws: WebSocket):
    await ws.accept()
    clients.add(ws)

    async def sender():
        while True:
            typ, payload = ws_queue.get()
            if typ == "audio":
                await ws.send_bytes(payload)
            else:
                await ws.send_json({"type": typ, "payload": payload})

    task = asyncio.create_task(sender())
    try:
        while True:
            data = await ws.receive_bytes()
            pcm  = np.frombuffer(data, np.int16)
            sr16 = resampy.resample(pcm.astype(np.float32), 48000, 16000)
            to_llm.put(sr16.astype(np.int16).tobytes())
    except WebSocketDisconnect:
        clients.remove(ws)
        task.cancel()

@app.get("/")
def index():
    return FileResponse("index.html")

# ─── BACKGROUND WORKERS ───────────────────────────────────────────────────────
def stt_worker():
    vad = webrtcvad.Vad(2)
    whisper = Model(WHISPER_MODEL_NAME)
    buff = b""
    frameSz = int(16000 * 0.03) * 2
    while True:
        chunk = to_llm.get()
        buff += chunk
        while len(buff) >= frameSz:
            frm, buff = buff[:frameSz], buff[frameSz:]
            if vad.is_speech(frm, 16000):
                audio = np.frombuffer(frm, np.int16).astype(np.float32) / 32768.0
                txt = "".join(seg.text for seg in whisper.transcribe(audio)).strip()
                if txt:
                    ws_queue.put(("text", txt))
                    text_q.put(txt)

def llm_worker():
    client = ollama.Client(host=OLLAMA_HOST)
    conv = [{"role":"system","content":DIA_SYSTEM_PROMPT}]
    while True:
        user_txt = text_q.get()
        conv.append({"role":"user","content":user_txt})
        full = ""
        for chunk in client.chat(model=OLLAMA_MODEL, messages=conv, stream=True):
            full += chunk["message"]["content"]
        ws_queue.put(("ai_response", full))
        conv.append({"role":"assistant","content":full})
        for sent in full.split("."):
            if sent.strip():
                to_tts.put(sent.strip())

def tts_worker():
    set_deterministic_seed(TTS_SEED)
    # reuse same config
    config = AutoConfig.from_pretrained(DIA_MODEL_NAME)
    dia = Dia.from_pretrained(
        DIA_MODEL_NAME,
        config=config,
        device=DEVICE,
        compute_dtype="float16"
    )
    ws_queue.put(("tts_status", "Dia TTS ready"))
    while True:
        text = to_tts.get()
        set_deterministic_seed(TTS_SEED)
        pcm24 = dia.generate(f"{FIXED_VOICE_PROMPT} {text}", use_torch_compile=False, verbose=False)
        pcm48 = resampy.resample(pcm24.astype(np.float32), 24000, 48000)
        pcm16 = (np.clip(pcm48, -0.9, 0.9) * 32767).astype(np.int16).tobytes()
        ws_queue.put(("audio", pcm16))

# ─── ENTRYPOINT ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DiaChat Voice AI")
    parser.add_argument("--skip-checks", action="store_true")
    args = parser.parse_args()

    if not args.skip_checks:
        run_preflight_checks()

    for fn in (stt_worker, llm_worker, tts_worker):
        threading.Thread(target=fn, daemon=True).start()

    print("🚀 Server running at http://0.0.0.0:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)
