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

# ---------- configuration ----------
MODELS_DIR = os.getenv("MODELS_DIR", "./models")
# Set Hugging Face cache directory to match setup_models.py
os.environ["HF_HUB_CACHE"] = os.path.abspath(MODELS_DIR)

WHISPER_MODEL_NAME = "base.en"
DIA_MODEL_NAME = "nari-labs/Dia-1.6B"
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:latest")

# Fixed seed for consistent voice generation
TTS_SEED = 42
FIXED_VOICE_PROMPT = "[S1]"  # Only use S1 speaker for consistency

DIA_SYSTEM_PROMPT = """You are a helpful and conversational AI assistant. Your responses will be converted into speech by an advanced Text-to-Speech (TTS) system. To make the speech sound more natural, please keep your responses coherent, directly answer the user's query, and maintain a friendly conversational tone. Keep responses moderate in length for optimal audio quality."""

# Auto-detect device (CUDA if available, otherwise CPU)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[main] Using device: {DEVICE}")

# ---------- Utility Functions ----------
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

# Set deterministic seed at startup
set_deterministic_seed(TTS_SEED)

# ---------- Pre-flight checks ----------

def check_ollama():
    """Test Ollama connectivity and model availability"""
    print("[check] Testing Ollama connection...")
    try:
        # Create client with host
        client = ollama.Client(host=OLLAMA_HOST)
        
        # Test basic connectivity
        response = client.list()
        models_list = response.get('models', [])
        available_models = []
        for model in models_list:
            if hasattr(model, 'model'):
                available_models.append(model.model)
            elif isinstance(model, dict) and 'name' in model:
                available_models.append(model['name'])
            else:
                print(f"[check] Warning: Unexpected model format: {model}")
        
        print(f"[check] ✅ Ollama connected. Available models: {available_models}")
        
        # Check if our required model is available
        if OLLAMA_MODEL not in available_models:
            print(f"[check] ⚠️ Model '{OLLAMA_MODEL}' not found. Attempting to pull...")
            try:
                client.pull(OLLAMA_MODEL)
                print(f"[check] ✅ Successfully pulled {OLLAMA_MODEL}")
            except Exception as e:
                print(f"[check] ❌ Failed to pull {OLLAMA_MODEL}: {e}")
                return False
        
        # Test model inference
        print(f"[check] Testing {OLLAMA_MODEL} inference...")
        test_response = client.chat(
            model=OLLAMA_MODEL,
            messages=[{"role": "user", "content": "Hello, are you working?"}]
        )
        response_text = test_response['message']['content']
        print(f"[check] ✅ Ollama test successful. Response: {response_text[:50]}...")
        return True
        
    except Exception as e:
        print(f"[check] ❌ Ollama connection failed: {e}")
        print(f"[check] Make sure Ollama is running at {OLLAMA_HOST}")
        return False

def check_whisper():
    """Test Whisper model loading"""
    print("[check] Testing Whisper model...")
    try:
        model = Model(WHISPER_MODEL_NAME)
        print(f"[check] ✅ Whisper model '{WHISPER_MODEL_NAME}' loaded successfully")
        
        # Test with a small audio sample (1 second of silence)
        test_audio = np.zeros(16000, dtype=np.float32)  # 1 second of silence at 16kHz
        segments = model.transcribe(test_audio)
        print(f"[check] ✅ Whisper inference test successful")
        return True
        
    except Exception as e:
        print(f"[check] ❌ Whisper model failed: {e}")
        print(f"[check] Make sure the model is downloaded manually")
        return False

def check_dia():
    """Test Dia model loading and inference"""
    print("[check] Testing Dia model...")
    try:
        # Set seed before loading model
        set_deterministic_seed(TTS_SEED)
        
        dia = Dia.from_pretrained(DIA_MODEL_NAME, device=DEVICE, compute_dtype="float16")
        print(f"[check] ✅ Dia model loaded successfully on {DEVICE}")
        
        # Test with a short text using fixed voice prompt
        print("[check] Testing Dia inference with fixed voice...")
        test_text = f"{FIXED_VOICE_PROMPT} Hello, this is a voice consistency test."
        
        # Set seed again before generation
        set_deterministic_seed(TTS_SEED)
        audio_output = dia.generate(test_text, use_torch_compile=False)
        print(f"[check] ✅ Dia inference successful. Generated {audio_output.shape} audio samples")
        return True
        
    except Exception as e:
        print(f"[check] ❌ Dia model failed: {e}")
        print(f"[check] Make sure you're logged into Hugging Face and the model is cached")
        return False

def check_audio_processing():
    """Test audio processing components"""
    print("[check] Testing audio processing...")
    try:
        # Test webrtcvad
        vad = webrtcvad.Vad(2)
        test_frame = np.zeros(480, dtype=np.int16).tobytes()  # 30ms frame at 16kHz
        vad.is_speech(test_frame, 16000)
        print("[check] ✅ WebRTC VAD working")
        
        # Test resampling
        test_audio = np.random.random(4096).astype(np.float32)
        resampled = resampy.resample(test_audio, 48000, 16000)
        print("[check] ✅ Audio resampling working")
        
        return True
        
    except Exception as e:
        print(f"[check] ❌ Audio processing failed: {e}")
        return False

def run_preflight_checks():
    """Run all pre-flight checks"""
    print("\n🚀 Starting DiaChat pre-flight checks...\n")
    
    checks = [
        ("Audio Processing", check_audio_processing),
        ("Whisper Model", check_whisper),
        ("Dia Model", check_dia),
        ("Ollama Service", check_ollama),
    ]
    
    results = {}
    for name, check_func in checks:
        print(f"\n{'='*50}")
        print(f"Checking: {name}")
        print('='*50)
        results[name] = check_func()
        time.sleep(0.5)  # Small delay between checks
    
    # Summary
    print(f"\n{'='*50}")
    print("PRE-FLIGHT RESULTS SUMMARY")
    print('='*50)
    
    all_passed = True
    for name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{name:20} {status}")
        if not passed:
            all_passed = False
    
    print('='*50)
    
    if all_passed:
        print("🎉 All checks passed! DiaChat is ready to start.\n")
        return True
    else:
        print("❌ Some checks failed. Please fix the issues above before starting DiaChat.")
        print("💡 Make sure models are downloaded manually as requested.\n")
        return False

# ---------- FastAPI ----------
app = FastAPI()
clients = set()

# Queues for threading
to_llm = queue.Queue(maxsize=10)
text_queue = queue.Queue(maxsize=10)
to_tts = queue.Queue(maxsize=10)
ws_out_queue = queue.Queue(maxsize=50)

@app.websocket("/ws/audio")
async def audio_ws(ws: WebSocket):
    await ws.accept()
    clients.add(ws)
    
    # Start WebSocket message handler for this client
    async def handle_outbound_messages():
        while True:
            try:
                # Check for messages to send to this client
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
    
    # Start the message handler task
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

# ---------- background workers ----------

def stt_worker():
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
    
    ws_out_queue.put(("whisper_status", "Whisper model loaded and ready", False))
    
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
                        ws_out_queue.put(("whisper_status", f"🎤 Capturing speech... ({audio_duration:.1f}s)", False))
                        last_status_time = current_time
                
                else:
                    silence_count += 1
                
                if speech_frames and silence_count > 10:
                    speech_audio = b"".join(speech_frames)
                    audio_duration = len(speech_frames) * FRAME_DURATION_MS / 1000
                    
                    audio_data = np.frombuffer(speech_audio, dtype=np.int16).astype(np.float32) / 32768.0
                    
                    if len(audio_data) > RATE * 0.5:
                        ws_out_queue.put(("whisper_status", f"🔄 Processing {audio_duration:.1f}s of audio...", True))
                        
                        segments = model.transcribe(audio_data)
                        txt = " ".join([segment.text for segment in segments]).strip()
                        
                        if txt and len(txt) > 2:
                            ws_out_queue.put(("whisper_status", f"✅ Transcribed: '{txt}'", False))
                            ws_out_queue.put(("text", txt))
                            text_queue.put(txt)
                        else:
                            ws_out_queue.put(("whisper_status", "⚠️ No clear speech detected", False))
                    else:
                        ws_out_queue.put(("whisper_status", f"⚠️ Audio too short ({audio_duration:.1f}s), ignoring", False))
                    
                    speech_frames = []
                    threading.Timer(0.5, lambda: ws_out_queue.put(("whisper_status", "👂 Listening for speech...", False))).start()
                    last_status_time = current_time
                    
            except Exception as e:
                print(f"[stt] Error processing frame: {e}")
                ws_out_queue.put(("whisper_status", f"❌ Error: {str(e)}", False))
                continue

def llm_worker():
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
        else:
            messages.append({"role": "assistant", "content": "(No audible response)"})

def tts_worker():
    # Set seed before loading model for consistency
    set_deterministic_seed(TTS_SEED)
    
    dia = Dia.from_pretrained(DIA_MODEL_NAME, device=DEVICE, compute_dtype="float16")
    ws_out_queue.put(("tts_status", "🤖 Dia TTS model loaded with consistent voice", False))
    
    while True:
        sentence = to_tts.get()
        if not sentence:
            continue
        
        # Use only S1 speaker for consistency and add period if missing
        if not sentence.endswith(('.', '!', '?')):
            sentence += '.'
            
        dia_input_text = f"{FIXED_VOICE_PROMPT} {sentence}"
        
        ws_out_queue.put(("tts_status", f"🔄 Generating consistent voice audio...", True))
        
        try:
            # Set seed before each generation for voice consistency
            set_deterministic_seed(TTS_SEED)
            
            # Generate audio with consistent settings
            pcm24 = dia.generate(
                dia_input_text, 
                use_torch_compile=False,  # Disable for consistency
                max_length=3000  # Limit length to prevent speed issues
            )
            
            if hasattr(pcm24, 'cpu'): 
                pcm24 = pcm24.cpu().numpy()
            elif not isinstance(pcm24, np.ndarray):
                pcm24 = np.array(pcm24) 

            # Resample to 48kHz for web audio
            pcm48 = resampy.resample(pcm24.astype(np.float32), 24000, 48000)
            
            # Normalize audio to prevent clipping and reduce noise
            pcm48 = np.clip(pcm48, -0.95, 0.95)
            pcm48_i16 = (pcm48 * 32767).astype(np.int16).tobytes()
            
            ws_out_queue.put(("tts_status", f"✅ Consistent voice audio generated", False))
            ws_out_queue.put(("audio", pcm48_i16))
            
        except Exception as e:
            print(f"[tts_worker] Error generating audio: {e}")
            ws_out_queue.put(("tts_status", f"❌ Error generating audio: {e}", False))
        
        threading.Timer(0.5, lambda: ws_out_queue.put(("tts_status", "👂 Ready for next text...", False))).start()

# ---------- routes ----------
@app.get("/")
def index():
    return FileResponse("index.html")

# ---------- main ----------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DiaChat - Real-time voice chat with AI")
    parser.add_argument(
        "--skip-checks", 
        action="store_true", 
        help="Skip pre-flight checks (faster startup, use when everything is already working)"
    )
    args = parser.parse_args()
    
    # Run pre-flight checks (unless skipped)
    if not args.skip_checks:
        if not run_preflight_checks():
            print("❌ Pre-flight checks failed. Exiting.")
            sys.exit(1)
    else:
        print("⚡ Skipping pre-flight checks for faster startup...")
    
    # Start background threads
    print("[main] Starting background workers...")
    for worker in (stt_worker, llm_worker, tts_worker):
        threading.Thread(target=worker, daemon=True).start()
    
    print(f"[main] Starting HTTP server on port 8000...")
    print(f"[main] Access via: http://localhost:8000")
    
    # Run HTTP server (no SSL for RunPod)
    uvicorn.run(app, host="0.0.0.0", port=8000)
