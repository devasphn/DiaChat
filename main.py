import os
import asyncio
import json
import queue
import threading
import ssl
import sys
import time
import argparse
import subprocess
import shutil
import random

import numpy as np
import resampy
import webrtcvad
import httpx
import ollama
from pywhispercpp.model import Model
from dia.model import Dia
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
import uvicorn
import torch

# ---------- ENHANCED CONFIGURATION ----------
MODELS_DIR = os.getenv("MODELS_DIR", "./models")
os.environ["HF_HUB_CACHE"] = os.path.abspath(MODELS_DIR)

WHISPER_MODEL_NAME = "base.en"
# Use the official Dia model from HuggingFace Hub
DIA_MODEL_NAME = "nari-labs/Dia-1.6B"
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:latest")

# VOICE CONSISTENCY SETTINGS
TTS_SEED = 42
FIXED_VOICE_PROMPT = "[S1]"

DIA_SYSTEM_PROMPT = """You are a helpful and conversational AI assistant. Your responses will be converted into speech by an advanced Text-to-Speech system. Keep your responses coherent, direct, and maintain a friendly conversational tone. Keep responses moderate in length for optimal audio quality."""

# Auto-detect device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[main] Using device: {DEVICE}")
print(f"[main] PyTorch version: {torch.__version__}")
print(f"[main] NumPy version: {np.__version__}")

# ---------- VOICE CONSISTENCY FUNCTIONS ----------
def set_deterministic_seed(seed: int):
    """Set deterministic seed for reproducible voice generation"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    print(f"[voice] Set deterministic seed: {seed}")

# Set seed immediately at module load
set_deterministic_seed(TTS_SEED)

# ---------- ENHANCED PRE-FLIGHT CHECKS ----------
def check_ollama():
    """Test Ollama connectivity and model availability"""
    print("[check] Testing Ollama connection...")
    try:
        client = ollama.Client(host=OLLAMA_HOST)
        response = client.list()
        models_list = response.get('models', [])
        available_models = []
        for model in models_list:
            if hasattr(model, 'model'):
                available_models.append(model.model)
            elif isinstance(model, dict) and 'name' in model:
                available_models.append(model['name'])
        
        print(f"[check] ✅ Ollama connected. Available models: {available_models}")
        
        if OLLAMA_MODEL not in available_models:
            print(f"[check] ⚠️ Model '{OLLAMA_MODEL}' not found. Attempting to pull...")
            try:
                client.pull(OLLAMA_MODEL)
                print(f"[check] ✅ Successfully pulled {OLLAMA_MODEL}")
            except Exception as e:
                print(f"[check] ❌ Failed to pull {OLLAMA_MODEL}: {e}")
                return False
        
        test_response = client.chat(
            model=OLLAMA_MODEL,
            messages=[{"role": "user", "content": "Hello, test message."}]
        )
        print(f"[check] ✅ Ollama test successful")
        return True
        
    except Exception as e:
        print(f"[check] ❌ Ollama connection failed: {e}")
        return False

def check_whisper():
    """Test Whisper model loading"""
    print("[check] Testing Whisper model...")
    try:
        model = Model(WHISPER_MODEL_NAME)
        print(f"[check] ✅ Whisper model '{WHISPER_MODEL_NAME}' loaded successfully")
        
        test_audio = np.zeros(16000, dtype=np.float32)
        segments = model.transcribe(test_audio)
        print(f"[check] ✅ Whisper inference test successful")
        return True
        
    except Exception as e:
        print(f"[check] ❌ Whisper model failed: {e}")
        return False

def check_dia():
    """Test Dia model loading with voice consistency - CORRECTED for PyTorch 2.7 + NumPy 2.2.6"""
    print("[check] Testing Dia model with voice consistency...")
    print(f"[check] Loading Dia model: {DIA_MODEL_NAME}")
    try:
        # Apply seed before model loading
        set_deterministic_seed(TTS_SEED)
        
        # Load Dia model - compatible with PyTorch 2.7 and NumPy 2.2.6
        dia = Dia.from_pretrained(DIA_MODEL_NAME, device=DEVICE, compute_dtype="float16")
        print(f"[check] ✅ Dia model loaded successfully on {DEVICE}")
        
        # Test voice generation with correct API
        print("[check] Testing voice consistency with fixed S1 speaker...")
        test_text = f"{FIXED_VOICE_PROMPT} Hello, this is a consistent voice test using fixed seed."
        
        # Apply seed again before generation
        set_deterministic_seed(TTS_SEED)
        
        # Use correct Dia API (compatible with latest version)
        audio_output = dia.generate(test_text, use_torch_compile=False, verbose=True)
        print(f"[check] ✅ Dia voice consistency test successful. Generated {audio_output.shape} samples")
        return True
        
    except Exception as e:
        print(f"[check] ❌ Dia model failed: {e}")
        print(f"[check] ℹ️ Model name used: {DIA_MODEL_NAME}")
        print(f"[check] ℹ️ Device: {DEVICE}")
        return False

def check_audio_processing():
    """Test audio processing components"""
    print("[check] Testing audio processing...")
    try:
        vad = webrtcvad.Vad(2)
        test_frame = np.zeros(480, dtype=np.int16).tobytes()
        vad.is_speech(test_frame, 16000)
        print("[check] ✅ WebRTC VAD working")
        
        test_audio = np.random.random(4096).astype(np.float32)
        resampled = resampy.resample(test_audio, 48000, 16000)
        print("[check] ✅ Audio resampling working")
        
        return True
        
    except Exception as e:
        print(f"[check] ❌ Audio processing failed: {e}")
        return False

def run_preflight_checks():
    """Run all pre-flight checks"""
    print("\n🚀 Starting DiaChat Voice-Optimized Pre-flight Checks...\n")
    print(f"[system] PyTorch: {torch.__version__}")
    print(f"[system] NumPy: {np.__version__}")
    print(f"[system] CUDA Available: {torch.cuda.is_available()}")
    
    checks = [
        ("Audio Processing", check_audio_processing),
        ("Whisper Model", check_whisper),
        ("Dia Voice Model", check_dia),
        ("Ollama Service", check_ollama),
    ]
    
    results = {}
    for name, check_func in checks:
        print(f"\n{'='*50}")
        print(f"Checking: {name}")
        print('='*50)
        results[name] = check_func()
        time.sleep(0.5)
    
    print(f"\n{'='*50}")
    print("VOICE-OPTIMIZED RESULTS SUMMARY")
    print('='*50)
    
    all_passed = True
    for name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{name:20} {status}")
        if not passed:
            all_passed = False
    
    print('='*50)
    
    if all_passed:
        print("🎉 All voice-optimized checks passed! DiaChat ready with consistent voice.\n")
        return True
    else:
        print("❌ Some checks failed. Please fix the issues above.\n")
        return False

# ---------- FASTAPI APPLICATION ----------
app = FastAPI()
clients = set()

to_llm = queue.Queue(maxsize=10)
text_queue = queue.Queue(maxsize=10)
to_tts = queue.Queue(maxsize=10)
ws_out_queue = queue.Queue(maxsize=50)

@app.websocket("/ws/audio")
async def audio_ws(ws: WebSocket):
    await ws.accept()
    clients.add(ws)
    
    async def handle_outbound_messages():
        while True:
            try:
                while not ws_out_queue.empty():
                    try:
                        message_parts = ws_out_queue.get_nowait()
                        msg_type = message_parts[0]
                        payload = message_parts[1]
                        
                        if msg_type == "text":
                            await ws.send_json({"type": "text", "payload": payload})
                        elif msg_type == "whisper_status":
                            processing = message_parts[2] if len(message_parts) > 2 else False
                            await ws.send_json({"type": "whisper_status", "payload": payload, "processing": processing})
                        elif msg_type == "tts_status":
                            processing = message_parts[2] if len(message_parts) > 2 else False
                            await ws.send_json({"type": "tts_status", "payload": payload, "processing": processing})
                        elif msg_type == "ai_response":
                            await ws.send_json({"type": "ai_response", "payload": payload})
                        elif msg_type == "audio":
                            await ws.send_bytes(payload)
                    except queue.Empty:
                        break
                await asyncio.sleep(0.01)
            except Exception as e:
                print(f"[ws] Error handling outbound messages: {e}")
                break
    
    handler_task = asyncio.create_task(handle_outbound_messages())
    
    try:
        while True:
            msg = await ws.receive_bytes()
            pcm = np.frombuffer(msg, np.int16)
            pcm16 = resampy.resample(pcm.astype(np.float32), 48000, 16000)
            to_llm.put(pcm16.astype(np.int16).tobytes())
    except WebSocketDisconnect:
        clients.remove(ws)
        handler_task.cancel()

# ---------- BACKGROUND WORKERS ----------
def stt_worker():
    """Speech-to-text worker with enhanced audio processing"""
    RATE = 16000
    FRAME_DURATION_MS = 30
    FRAME_SIZE = int(RATE * FRAME_DURATION_MS / 1000)
    FRAME_BYTES = FRAME_SIZE * 2
    
    vad = webrtcvad.Vad(2)
    model = Model(WHISPER_MODEL_NAME)
    
    audio_buffer = b""
    speech_frames = []
    silence_count = 0
    last_status_time = 0
    
    ws_out_queue.put(("whisper_status", "Whisper ready for voice input", False))
    
    while True:
        buf = to_llm.get()
        audio_buffer += buf
        
        while len(audio_buffer) >= FRAME_BYTES:
            frame = audio_buffer[:FRAME_BYTES]
            audio_buffer = audio_buffer[FRAME_BYTES:]
            
            try:
                is_speech = vad.is_speech(frame, RATE)
                current_time = time.time()
                
                if is_speech:
                    speech_frames.append(frame)
                    silence_count = 0
                    
                    if current_time - last_status_time > 0.5:
                        audio_duration = len(speech_frames) * FRAME_DURATION_MS / 1000
                        ws_out_queue.put(("whisper_status", f"🎤 Recording speech... ({audio_duration:.1f}s)", False))
                        last_status_time = current_time
                
                else:
                    silence_count += 1
                
                if speech_frames and silence_count > 10:
                    speech_audio = b"".join(speech_frames)
                    audio_duration = len(speech_frames) * FRAME_DURATION_MS / 1000
                    
                    audio_data = np.frombuffer(speech_audio, dtype=np.int16).astype(np.float32) / 32768.0
                    
                    if len(audio_data) > RATE * 0.5:
                        ws_out_queue.put(("whisper_status", f"🔄 Processing {audio_duration:.1f}s of speech...", True))
                        
                        segments = model.transcribe(audio_data)
                        txt = " ".join([segment.text for segment in segments]).strip()
                        
                        if txt and len(txt) > 2:
                            ws_out_queue.put(("whisper_status", f"✅ Heard: '{txt}'", False))
                            ws_out_queue.put(("text", txt))
                            text_queue.put(txt)
                        else:
                            ws_out_queue.put(("whisper_status", "⚠️ No clear speech detected", False))
                    else:
                        ws_out_queue.put(("whisper_status", f"⚠️ Speech too short ({audio_duration:.1f}s)", False))
                    
                    speech_frames = []
                    threading.Timer(0.5, lambda: ws_out_queue.put(("whisper_status", "👂 Listening for voice...", False))).start()
                    last_status_time = current_time
                    
            except Exception as e:
                print(f"[stt] Error processing audio: {e}")
                continue

def llm_worker():
    """LLM worker for generating responses"""
    client = ollama.Client(host=OLLAMA_HOST)
    messages = []
    
    while True:
        user_part = text_queue.get()

        if not messages:
            messages.append({"role": "system", "content": DIA_SYSTEM_PROMPT})
        
        messages.append({"role": "user", "content": user_part})
        
        full_response_content = ""
        sentence_buffer = ""

        for chunk in client.chat(model=OLLAMA_MODEL, messages=messages, stream=True):
            token = chunk["message"]["content"]
            full_response_content += token
            sentence_buffer += token
            
            if any(sentence_buffer.strip().endswith(p) for p in ['.', '?', '!']):
                current_sentence_text = sentence_buffer.strip()
                if current_sentence_text:
                    to_tts.put(current_sentence_text)
                sentence_buffer = ""
        
        if sentence_buffer.strip():
            to_tts.put(sentence_buffer.strip())

        if full_response_content.strip():
            ws_out_queue.put(("ai_response", full_response_content.strip()))
            messages.append({"role": "assistant", "content": full_response_content.strip()})

def tts_worker():
    """TTS worker with CORRECTED Dia.generate() API calls for PyTorch 2.7 + NumPy 2.2.6"""
    # Set seed before loading model for voice consistency
    set_deterministic_seed(TTS_SEED)
    
    # Load Dia model compatible with PyTorch 2.7 and NumPy 2.2.6
    dia = Dia.from_pretrained(DIA_MODEL_NAME, device=DEVICE, compute_dtype="float16")
    ws_out_queue.put(("tts_status", "🎯 Voice-optimized Dia TTS ready (S1 speaker only)", False))
    
    while True:
        sentence = to_tts.get()
        if not sentence:
            continue
        
        # Ensure proper sentence ending
        if not sentence.endswith(('.', '!', '?')):
            sentence += '.'
            
        # Use ONLY S1 speaker for consistency
        dia_input_text = f"{FIXED_VOICE_PROMPT} {sentence}"
        
        ws_out_queue.put(("tts_status", f"🔄 Generating consistent S1 voice...", True))
        
        try:
            # CRITICAL: Reset seed before each generation for voice consistency
            set_deterministic_seed(TTS_SEED)
            
            # Use correct Dia API (compatible with latest PyTorch 2.7 + NumPy 2.2.6)
            pcm24 = dia.generate(dia_input_text, use_torch_compile=False, verbose=False)
            
            if hasattr(pcm24, 'cpu'): 
                pcm24 = pcm24.cpu().numpy()
            elif not isinstance(pcm24, np.ndarray):
                pcm24 = np.array(pcm24) 

            # Enhanced audio processing for better quality
            pcm48 = resampy.resample(pcm24.astype(np.float32), 24000, 48000)
            
            # Normalize and prevent clipping
            pcm48 = np.clip(pcm48, -0.9, 0.9)
            pcm48_i16 = (pcm48 * 32767).astype(np.int16).tobytes()
            
            ws_out_queue.put(("tts_status", f"✅ Consistent S1 voice generated", False))
            ws_out_queue.put(("audio", pcm48_i16))
            
        except Exception as e:
            print(f"[tts] Error generating voice: {e}")
            ws_out_queue.put(("tts_status", f"❌ Voice generation error: {e}", False))
        
        threading.Timer(0.3, lambda: ws_out_queue.put(("tts_status", "🎯 Ready for next text (S1 voice)", False))).start()

# ---------- ROUTES ----------
@app.get("/")
def index():
    return FileResponse("index.html")

# ---------- MAIN ----------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DiaChat - Voice-Optimized Real-time AI Chat")
    parser.add_argument(
        "--skip-checks", 
        action="store_true", 
        help="Skip pre-flight checks (use when everything is working)"
    )
    args = parser.parse_args()
    
    # Run voice-optimized pre-flight checks
    if not args.skip_checks:
        if not run_preflight_checks():
            print("❌ Voice-optimized checks failed. Exiting.")
            sys.exit(1)
    else:
        print("⚡ Skipping checks for faster startup...")
    
    # Start background workers
    print("[main] Starting voice-optimized workers...")
    for worker in (stt_worker, llm_worker, tts_worker):
        threading.Thread(target=worker, daemon=True).start()
    
    print(f"[main] 🎯 Voice-optimized DiaChat server starting on port 8000...")
    print(f"[main] Access: http://localhost:8000")
    
    uvicorn.run(app, host="0.0.0.0", port=8000)
