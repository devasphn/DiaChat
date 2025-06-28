import os
import sys
import argparse
import asyncio
import json
import queue
import threading
import time
import random

import numpy as np
import resampy
import webrtcvad
import httpx
import ollama
import torch
from pywhispercpp.model import Model
from dia.model import Dia
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
import uvicorn

# ---------- CONFIGURATION ----------
MODELS_DIR = os.getenv("MODELS_DIR", "./models")
os.environ["HF_HUB_CACHE"] = MODELS_DIR

WHISPER_MODEL_NAME = "base.en"
DIA_MODEL_NAME = "nari-labs/Dia-1.6B"
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:latest")

TTS_SEED = 42
FIXED_VOICE_PROMPT = "[S1]"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[main] Device: {DEVICE}")

def set_deterministic_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    print(f"[voice] Seed set to {seed}")

set_deterministic_seed(TTS_SEED)

# ---------- PRE-FLIGHT CHECKS ----------
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
    set_deterministic_seed(TTS_SEED)
    dia = Dia.from_pretrained(DIA_MODEL_NAME, device=DEVICE, compute_dtype="float16")
    print("[check] ✅ Dia loaded")
    set_deterministic_seed(TTS_SEED)
    audio = dia.generate(f"{FIXED_VOICE_PROMPT} Test", use_torch_compile=False, verbose=False)
    print(f"[check] ✅ Dia generated {audio.shape[0]} samples")
    return True

def check_ollama():
    print("[check] Ollama service...")
    client = ollama.Client(host=OLLAMA_HOST)
    models = [m.model if hasattr(m, "model") else m["name"] for m in client.list().get("models", [])]
    print(f"[check] Available: {models}")
    if OLLAMA_MODEL not in models:
        client.pull(OLLAMA_MODEL)
    client.chat(model=OLLAMA_MODEL, messages=[{"role":"user","content":"ping"}])
    print("[check] ✅ Ollama OK")
    return True

def run_preflight_checks():
    print("\n🚀 Pre-flight checks:\n")
    for name, fn in [
        ("Audio", check_audio),
        ("Whisper", check_whisper),
        ("Dia TTS", check_dia),
        ("Ollama", check_ollama),
    ]:
        print(f"--- {name} ---")
        if not fn():
            print(f"❌ {name} failed")
            sys.exit(1)
    print("\n🎉 All checks passed!\n")

# ---------- FASTAPI & WEBSOCKET ----------
app = FastAPI()
clients = set()
to_llm = queue.Queue()
text_queue = queue.Queue()
to_tts = queue.Queue()
ws_out_queue = queue.Queue()

@app.websocket("/ws/audio")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    clients.add(ws)

    async def sender():
        while True:
            try:
                msg_type, payload = ws_out_queue.get_nowait()
            except queue.Empty:
                await asyncio.sleep(0.01)
                continue
            if msg_type == "audio":
                await ws.send_bytes(payload)
            else:
                await ws.send_json({"type": msg_type, "payload": payload})

    send_task = asyncio.create_task(sender())

    try:
        while True:
            data = await ws.receive_bytes()
            pcm = np.frombuffer(data, np.int16)
            resampled = resampy.resample(pcm.astype(np.float32), 48000, 16000)
            to_llm.put(resampled.astype(np.int16).tobytes())
    except WebSocketDisconnect:
        clients.remove(ws)
        send_task.cancel()

@app.get("/")
def get_index():
    return FileResponse("index.html")

# ---------- BACKGROUND WORKERS ----------
def stt_worker():
    vad = webrtcvad.Vad(2)
    whisper = Model(WHISPER_MODEL_NAME)
    buffer = b""
    FRAME_SIZE = int(16000 * 0.03) * 2
    while True:
        data = to_llm.get()
        buffer += data
        while len(buffer) >= FRAME_SIZE:
            frame, buffer = buffer[:FRAME_SIZE], buffer[FRAME_SIZE:]
            if vad.is_speech(frame, 16000):
                audio = np.frombuffer(frame, np.int16).astype(np.float32) / 32768.0
                segments = whisper.transcribe(audio)
                text = "".join(seg.text for seg in segments).strip()
                if text:
                    ws_out_queue.put(("text", text))
                    text_queue.put(text)

def llm_worker():
    client = ollama.Client(host=OLLAMA_HOST)
    convo = [{"role":"system","content":DIA_SYSTEM_PROMPT}]
    while True:
        user_text = text_queue.get()
        convo.append({"role":"user","content":user_text})
        full = ""
        for chunk in client.chat(model=OLLAMA_MODEL, messages=convo, stream=True):
            token = chunk["message"]["content"]
            full += token
        ws_out_queue.put(("ai_response", full))
        convo.append({"role":"assistant","content":full})
        for sent in full.split("."):
            if sent.strip():
                to_tts.put(sent.strip())

def tts_worker():
    set_deterministic_seed(TTS_SEED)
    dia = Dia.from_pretrained(DIA_MODEL_NAME, device=DEVICE, compute_dtype="float16")
    ws_out_queue.put(("tts_status", "Dia TTS ready"))
    while True:
        text = to_tts.get()
        set_deterministic_seed(TTS_SEED)
        audio = dia.generate(f"{FIXED_VOICE_PROMPT} {text}", use_torch_compile=False, verbose=False)
        resampled = resampy.resample(audio.astype(np.float32), 24000, 48000)
        pcm16 = (np.clip(resampled, -0.9, 0.9) * 32767).astype(np.int16).tobytes()
        ws_out_queue.put(("audio", pcm16))

# ---------- MAIN ----------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DiaChat Voice AI")
    parser.add_argument("--skip-checks", action="store_true")
    args = parser.parse_args()

    if not args.skip_checks:
        run_preflight_checks()

    for worker in (stt_worker, llm_worker, tts_worker):
        threading.Thread(target=worker, daemon=True).start()

    print("Starting server on port 8000...")
    uvicorn.run(app, host="0.0.0.0", port=8000)
