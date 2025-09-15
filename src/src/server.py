import asyncio
import uuid
from contextlib import asynccontextmanager
import os
import numpy as np
from collections import deque # For efficient audio buffering
from huggingface_hub import snapshot_download
import wave # For TTS audio processing (server-side, if saving chunks there)
import pyaudio # For CoquiEngine stream info
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.websockets import WebSocketState # Import WebSocketState

from vllm import AsyncLLMEngine, AsyncEngineArgs, SamplingParams

# --- Faster Whisper Dependencies ---
from faster_whisper import WhisperModel

# --- Tokenizer for History Management ---
from transformers import AutoTokenizer

# --- Silero VAD Dependencies ---
import torch
# Ensure you have torchaudio installed: pip install torchaudio
from SmartAITool.core import cprint # Custom print function for colored output
# --- RealtimeTTS Dependencies ---
import RealtimeTTS
from RealtimeTTS import TextToAudioStream, CoquiEngine

# --- Configuration ---
# vLLM Model Configurationpyth
HF_MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct-AWQ" # Example, use your desired model
LOCAL_MODEL_DIR = f"/home/ubuntu/ai_service/checkpoints/local_models/{HF_MODEL_ID}"

MAX_TOKENS = 512
TEMPERATURE = 0.7
DTYPE = "float16" # or "bfloat16" or "auto"
QUANTIZATION = "awq_marlin" # or "awq", "gptq", None etc.

# Faster Whisper Model Configuration
WHISPER_MODEL_SIZE = "small" # "tiny", "base", "small", "medium", "large-v2", "large-v3"
WHISPER_DEVICE = "cuda" # "cuda" or "cpu"
WHISPER_COMPUTE_TYPE = "float16" # "float16", "int8_float16", "int8"

# Chat History Configuration
MAX_CONTEXT_TOKENS = 8192

# Silero VAD Configuration
VAD_SAMPLE_RATE = 16000 # Must match incoming audio
VAD_THRESHOLD = 0.5
MIN_SILENCE_DURATION_MS = 700
SPEECH_PAD_MS = 200
VAD_FRAME_SIZE_MS = 32
VAD_FRAME_SIZE_SAMPLES = int(VAD_FRAME_SIZE_MS * VAD_SAMPLE_RATE / 1000)

# RealtimeTTS CoquiEngine Configuration
TTS_REFERENCE_VOICE_WAV = "my_reference_voice.wav" # Path to your reference speaker WAV
TTS_COQUI_MODELS_PATH = '/home/ubuntu/ai_service/checkpoints/xtts' # Or path to your local Coqui TTS models directory
TTS_LANGUAGE = "en"
# These must match what the client expects for saving WAV files
TTS_TARGET_SAMPLE_RATE = 24000
TTS_TARGET_CHANNELS = 1
TTS_TARGET_SAMPLE_WIDTH_BYTES = 2 # 2 bytes for int16

# --- Global Variables ---
llm_engine: AsyncLLMEngine = None
whisper_model: WhisperModel = None
history_tokenizer: AutoTokenizer = None
vad_model = None
# tts_engine: CoquiEngine = None
# tts_stream: TextToAudioStream = None


async def download_model_if_not_exists(model_id: str, local_dir: str):
    """Downloads a Hugging Face model snapshot if it doesn't exist locally."""
    if os.path.isdir(local_dir) and os.listdir(local_dir):
        print(f"Model already exists in '{local_dir}'. Skipping download.")
        return True
    
    print(f"Model not found in '{local_dir}'. Downloading '{model_id}'...")
    os.makedirs(os.path.dirname(local_dir), exist_ok=True)
    os.makedirs(local_dir, exist_ok=True)
    try:
        snapshot_download(repo_id=model_id, local_dir=local_dir, local_dir_use_symlinks=False)
        print(f"Model '{model_id}' downloaded successfully to '{local_dir}'.")
        return True
    except Exception as e:
        print(f"Failed to download model '{model_id}' to '{local_dir}': {e}")
        return False

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manages the lifecycle of models and resources for the FastAPI application."""
    global llm_engine, whisper_model, history_tokenizer, vad_model, tts_engine, tts_stream
    
    # 1. Download and Load vLLM Model
    download_success = await download_model_if_not_exists(HF_MODEL_ID, LOCAL_MODEL_DIR)
    if not download_success:
        cprint("[LOG] vLLM model download failed. vLLM engine will not start.", 'red')
    else:
        cprint(f"[LOG] Starting up: Loading vLLM model from local path '{LOCAL_MODEL_DIR}' with dtype='{DTYPE}' and quantization='{QUANTIZATION}'...", 'green')
        try:
            engine_args = AsyncEngineArgs(
                model=LOCAL_MODEL_DIR,
                dtype=DTYPE,
                quantization=QUANTIZATION,
                gpu_memory_utilization=0.2,  # Adjust as needed
            )
            llm_engine = AsyncLLMEngine.from_engine_args(engine_args)
            cprint("[LOG] vLLM model loaded successfully.", 'green')
        except Exception as e:
            cprint(f"[LOG] Failed to load vLLM engine from '{LOCAL_MODEL_DIR}': {e}", 'red')
            llm_engine = None 

    # 2. Load Faster Whisper Model
    cprint(f"[LOG] Loading Faster Whisper model '{WHISPER_MODEL_SIZE}' on '{WHISPER_DEVICE}' with compute_type='{WHISPER_COMPUTE_TYPE}'...", 'green')
    try:
        whisper_model = WhisperModel(WHISPER_MODEL_SIZE, device=WHISPER_DEVICE, compute_type=WHISPER_COMPUTE_TYPE)
        cprint("[LOG] Faster Whisper model loaded successfully.", 'green')
    except Exception as e:
        cprint(f"[LOG] Failed to load Faster Whisper model: {e}", 'red')
        whisper_model = None

    # 3. Load Tokenizer for History Management
    cprint(f"[LOG] Loading tokenizer for history management from '{LOCAL_MODEL_DIR}'...", 'green')
    try:
        history_tokenizer = AutoTokenizer.from_pretrained(LOCAL_MODEL_DIR, trust_remote_code=True)
        if history_tokenizer.chat_template is None:
            cprint("[LOG] Warning: Model has no chat template. Using a default one for history management.", 'yellow')
            history_tokenizer.chat_template = (
                "{% for message in messages %}"
                "{% if message['role'] == 'user' %}"
                "{{ '<|user|>\n' + message['content'] + eos_token }}"
                "{% elif message['role'] == 'assistant' %}"
                "{{ '<|assistant|>\n' + message['content'] + eos_token }}"
                "{% elif message['role'] == 'system' %}"
                "{{ '<|system|>\n' + message['content'] + eos_token }}"
                "{% endif %}"
                "{% endfor %}"
                "{% if add_generation_prompt %}{{ '<|assistant|>\n' }}{% endif %}"
            )
        cprint("[LOG] History tokenizer loaded successfully.", 'green')
    except Exception as e:
        cprint(f"[LOG] Failed to load history tokenizer: {e}", 'red')
        history_tokenizer = None

    # 4. Load Silero VAD Model
    cprint(f"[LOG] Loading Silero VAD model on '{WHISPER_DEVICE}'...", 'green')
    try:
        vad_model_torch, _ = torch.hub.load(repo_or_dir='snakers4/silero-vad',
                                        model='silero_vad',
                                        force_reload=False,
                                        trust_repo=True)
        vad_model = vad_model_torch.to(WHISPER_DEVICE)
        cprint("[LOG] Silero VAD model loaded successfully.", 'green')
    except Exception as e:
        cprint(f"[LOG] Failed to load Silero VAD model: {e}", 'red')
        vad_model = None

    # 5. Initialize RealtimeTTS CoquiEngine and TextToAudioStream
    cprint("[LOG] Initializing RealtimeTTS CoquiEngine...", 'green')
    try:
        coqui_engine_params = {
            "model_name": "tts_models/multilingual/multi-dataset/xtts_v2",
            "language": TTS_LANGUAGE,
            "local_models_path": TTS_COQUI_MODELS_PATH,
        }
        if os.path.exists(TTS_REFERENCE_VOICE_WAV):
            coqui_engine_params["voice"] = TTS_REFERENCE_VOICE_WAV
            cprint(f"[LOG] Using reference voice for TTS: {TTS_REFERENCE_VOICE_WAV}", 'green')
        else:
            cprint(f"[LOG] Reference voice '{TTS_REFERENCE_VOICE_WAV}' not found. Coqui TTS will use a default voice.", 'yellow')

        tts_engine = CoquiEngine(**coqui_engine_params)
        tts_stream = TextToAudioStream(tts_engine, muted=True, log_characters=False)
        cprint("[LOG] RealtimeTTS CoquiEngine and TextToAudioStream initialized successfully.", 'green')
    except Exception as e:
        cprint(f"[LOG] Error initializing RealtimeTTS CoquiEngine: {e}", 'red')
        tts_engine = None
        tts_stream = None

    yield 

    cprint("[LOG] Shutting down.", 'yellow')

app = FastAPI(lifespan=lifespan)

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print(f"WebSocket connected: {websocket.client}")

    chat_history = []
    chat_history.append({"role": "system", "content": "You are a helpful AI assistant. Respond concisely and accurately."})

    if llm_engine is None:
        if websocket.client_state == WebSocketState.CONNECTED:
            await websocket.send_text("[ERROR] vLLM engine not initialized. Server might be starting or failed to load model.")
        await websocket.close()
        return
    if whisper_model is None and vad_model is None: 
        print("[WARNING] Whisper or VAD model not loaded. Audio input will not be processed.")
    if tts_engine is None or tts_stream is None:
        print("[WARNING] TTS engine not loaded during startup. Audio output will not be generated by default.")


    audio_buffer = deque() 
    speech_start_idx = None 
    last_voice_activity_idx = 0 
    silence_frames_count = 0 
    total_samples_received = 0 

    try:
        while True:
            received_message_dict = await websocket.receive()
            user_input_content = None 

            if received_message_dict['type'] == 'websocket.disconnect':
                print(f"Client initiated disconnect: {websocket.client} with code {received_message_dict.get('code')}")
                if speech_start_idx is not None and whisper_model is not None and vad_model is not None:
                    print("VAD: Client disconnected mid-speech. Attempting to process remaining audio.")
                    await process_speech_segment(websocket, audio_buffer, speech_start_idx, last_voice_activity_idx, chat_history, total_samples_received)
                break 

            if received_message_dict['type'] == 'websocket.receive':
                if received_message_dict.get('text') is not None:
                    audio_buffer.clear()
                    speech_start_idx = None
                    last_voice_activity_idx = 0
                    silence_frames_count = 0
                    total_samples_received = 0

                    user_input_content = received_message_dict['text']
                    print(f"Received text prompt from {websocket.client}: '{user_input_content}'")
                    if websocket.client_state == WebSocketState.CONNECTED:
                        await websocket.send_text(f"[PROMPT] {user_input_content}")
                
                elif received_message_dict.get('bytes') is not None:
                    if vad_model is None or whisper_model is None:
                        if websocket.client_state == WebSocketState.CONNECTED:
                            await websocket.send_text("[ERROR] VAD or Faster Whisper model not loaded. Cannot process audio.")
                        continue

                    audio_bytes = received_message_dict['bytes']
                    new_audio_chunk_np_int16 = np.frombuffer(audio_bytes, dtype=np.int16)
                    
                    audio_buffer.extend(new_audio_chunk_np_int16)
                    
                    audio_chunk_float32 = new_audio_chunk_np_int16.astype(np.float32) / 32768.0
                    # Use WHISPER_DEVICE for VAD tensor as vad_model was moved to this device
                    audio_tensor = torch.from_numpy(audio_chunk_float32).unsqueeze(0).to(WHISPER_DEVICE) 


                    for i in range(0, audio_tensor.shape[1], VAD_FRAME_SIZE_SAMPLES):
                        frame = audio_tensor[:, i:i + VAD_FRAME_SIZE_SAMPLES]
                        if frame.shape[1] < VAD_FRAME_SIZE_SAMPLES:
                            break 

                        vad_score = vad_model(frame, VAD_SAMPLE_RATE).item()
                        current_frame_end_abs_idx = total_samples_received + i + VAD_FRAME_SIZE_SAMPLES

                        if vad_score >= VAD_THRESHOLD:
                            last_voice_activity_idx = current_frame_end_abs_idx
                            silence_frames_count = 0
                            if speech_start_idx is None:
                                speech_start_idx = current_frame_end_abs_idx - VAD_FRAME_SIZE_SAMPLES 
                                if websocket.client_state == WebSocketState.CONNECTED:
                                    await websocket.send_text("[VOICE_START]")
                                print(f"VAD: VOICE_START detected at {speech_start_idx / VAD_SAMPLE_RATE:.2f}s")
                        else: 
                            if speech_start_idx is not None: 
                                silence_frames_count += 1
                                current_silence_duration_ms = silence_frames_count * VAD_FRAME_SIZE_MS
                                if current_silence_duration_ms >= MIN_SILENCE_DURATION_MS:
                                    speech_end_idx = last_voice_activity_idx 
                                    print(f"VAD: VOICE_END detected at {speech_end_idx / VAD_SAMPLE_RATE:.2f}s. Processing segment.")
                                    
                                    transcribed_text = await process_speech_segment(
                                        websocket, audio_buffer, speech_start_idx, speech_end_idx, 
                                        chat_history, total_samples_received
                                    )
                                    user_input_content = transcribed_text 
                                    
                                    speech_start_idx = None
                                    last_voice_activity_idx = 0 
                                    silence_frames_count = 0
                                    if user_input_content: 
                                        break 
                    
                    total_samples_received += len(new_audio_chunk_np_int16)
                    if user_input_content: 
                        pass 
                    else: 
                        continue 

                else: 
                    print(f"Received 'websocket.receive' message without 'text' or 'bytes': {received_message_dict}")
                    if websocket.client_state == WebSocketState.CONNECTED:
                        await websocket.send_text("[ERROR] Invalid message format.")
                    continue
            else: 
                print(f"Received unhandled ASGI message type: {received_message_dict['type']}")
                continue 

            if user_input_content:
                await process_llm_and_tts_request(websocket, user_input_content, chat_history)
                user_input_content = None 

    except WebSocketDisconnect:
        print(f"WebSocket disconnected: {websocket.client}")
        if speech_start_idx is not None and whisper_model is not None and vad_model is not None:
            print("VAD: Client disconnected mid-speech during WebSocketDisconnect. Attempting to process remaining audio.")
            await process_speech_segment(websocket, audio_buffer, speech_start_idx, last_voice_activity_idx, chat_history, total_samples_received)
    except Exception as e:
        print(f"An unexpected error occurred with {websocket.client}: {e}")
        try:
            if websocket.client_state == WebSocketState.CONNECTED:
                await websocket.send_text(f"[ERROR] Server error: {str(e)}")
            await websocket.close(code=1011)
        except Exception as close_exc: 
            print(f"Error trying to close websocket: {close_exc}")
    finally:
        print(f"Connection handler for {websocket.client} finished.")


async def process_speech_segment(websocket: WebSocket, audio_buffer: deque, speech_start_abs_idx: int, speech_end_abs_idx: int, chat_history: list, current_total_samples_received: int) -> str | None:
    """ Extracts, pads, transcribes speech. Returns transcription or None. """
    if whisper_model is None:
        if websocket.client_state == WebSocketState.CONNECTED:
            await websocket.send_text("[ERROR] Whisper model not loaded. Cannot transcribe.")
        return None

    pad_samples = int(SPEECH_PAD_MS * VAD_SAMPLE_RATE / 1000)
    buffer_start_abs_idx = current_total_samples_received - len(audio_buffer)
    
    relative_start_idx = max(0, speech_start_abs_idx - buffer_start_abs_idx - pad_samples)
    relative_end_idx = min(len(audio_buffer), speech_end_abs_idx - buffer_start_abs_idx + pad_samples)

    current_full_buffer_np = np.array(list(audio_buffer), dtype=np.int16) 
    
    if relative_start_idx >= relative_end_idx:
        print("Warning: Calculated empty speech segment for ASR. Skipping.")
        if websocket.client_state == WebSocketState.CONNECTED:
            await websocket.send_text("[INFO] Empty audio segment for ASR.")
        samples_to_remove = min(relative_end_idx, len(audio_buffer))
        for _ in range(samples_to_remove):
            if audio_buffer: audio_buffer.popleft()
        print(f"Buffer cleared after empty segment. Remaining samples: {len(audio_buffer)}")
        return None

    speech_segment_np_int16 = current_full_buffer_np[relative_start_idx:relative_end_idx]
    audio_for_whisper = speech_segment_np_int16.astype(np.float32) / 32768.0

    print(f"Transcribing {len(audio_for_whisper)/VAD_SAMPLE_RATE:.2f}s of audio for ASR...")
    transcription = None
    try:
        segments, info = await asyncio.to_thread(
            whisper_model.transcribe,
            audio_for_whisper,
            beam_size=5,
            without_timestamps=True,
            vad_filter=False 
        )
        transcription = "".join(segment.text for segment in segments).strip()
        
        print(f"Transcription (Lang: {info.language}): '{transcription}'")
        if websocket.client_state == WebSocketState.CONNECTED:
            await websocket.send_text("[VOICE_END]") 
        
        if not transcription:
            if websocket.client_state == WebSocketState.CONNECTED:
                await websocket.send_text("[INFO] No speech transcribed from audio segment.")
        else:
            if websocket.client_state == WebSocketState.CONNECTED:
                await websocket.send_text(f"[TRANSCRIPTION] {transcription}")
            
    except Exception as e:
        print(f"Error during Faster Whisper transcription: {e}")
        if websocket.client_state == WebSocketState.CONNECTED:
            await websocket.send_text(f"[ERROR] Transcription failed: {e}")
    finally:
        samples_to_remove = min(relative_end_idx, len(audio_buffer))
        for _ in range(samples_to_remove):
            if audio_buffer: 
                audio_buffer.popleft()
        print(f"Buffer cleared. Remaining samples: {len(audio_buffer)}")
    
    return transcription


async def process_llm_and_tts_request(websocket: WebSocket, user_input_content: str, chat_history: list):
    """ Gets LLM response for user_input_content, then synthesizes TTS and streams audio. """
    if llm_engine is None:
        if websocket.client_state == WebSocketState.CONNECTED:
            await websocket.send_text("[ERROR] LLM Engine not available.")
        return

    chat_history.append({"role": "user", "content": user_input_content})
    final_prompt_for_llm = ""
    if history_tokenizer:
        try:
            temp_history = list(chat_history)
            current_tokens = len(history_tokenizer.apply_chat_template(temp_history, tokenize=True, add_generation_prompt=True))
            trim_index = 1 
            while (current_tokens + MAX_TOKENS > MAX_CONTEXT_TOKENS) and (len(temp_history) > 1): 
                if temp_history[trim_index]["role"] == "system": 
                    trim_index +=1
                    if trim_index >= len(temp_history): break
                    continue
                if trim_index + 1 < len(temp_history) and \
                   temp_history[trim_index]["role"] == "user" and \
                   temp_history[trim_index+1]["role"] == "assistant":
                    temp_history.pop(trim_index) 
                    temp_history.pop(trim_index) 
                else: 
                    temp_history.pop(trim_index)
                
                if not temp_history or trim_index >= len(temp_history) and len(temp_history) > 1 : 
                    trim_index = 1 if len(temp_history) > 1 else len(temp_history)
                current_tokens = len(history_tokenizer.apply_chat_template(temp_history, tokenize=True, add_generation_prompt=True))
                if trim_index >= len(temp_history) and len(temp_history) > 1: 
                    trim_index = 1
            final_prompt_for_llm = history_tokenizer.apply_chat_template(temp_history, tokenize=False, add_generation_prompt=True)
            if len(temp_history) < len(chat_history):
                 print(f"Chat history effectively trimmed for prompt from {len(chat_history)} to {len(temp_history)} messages for context length.")
            print(f"Using {len(temp_history)} messages for prompt. Final prompt (approx tokens): {current_tokens}")
        except Exception as e:
            print(f"Error during history tokenization/templating: {e}")
            final_prompt_for_llm = "\n".join([f"{m['role']}: {m['content']}" for m in chat_history[-5:]]) + "\nassistant:"
    else: 
        print("Warning: History tokenizer not loaded. Using simple history concatenation.")
        temp_history_slice = chat_history[-10:] 
        final_prompt_for_llm = "\n".join([f"{m['role']}: {m['content']}" for m in temp_history_slice]) + "\nassistant:"

    request_id = str(uuid.uuid4())
    sampling_params = SamplingParams(
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
    )

    cprint(f"LLM: Generating response for request_id: {request_id}...")
    full_llm_response_text = ""
    try:
        results_generator = llm_engine.generate(prompt=final_prompt_for_llm, sampling_params=sampling_params, request_id=request_id)
        final_output = None
        async for request_output in results_generator:
            final_output = request_output 
            if request_output.finished:
                full_llm_response_text = request_output.outputs[0].text.strip()
                # cprint(full_llm_response_text, 'green')
                break 
        
        if not full_llm_response_text and final_output: 
             full_llm_response_text = final_output.outputs[0].text.strip()

        if not full_llm_response_text:
            print(f"LLM Warning: Empty response for request_id: {request_id}")
            if websocket.client_state == WebSocketState.CONNECTED: 
                await websocket.send_text("[INFO] LLM returned an empty response.")
    
    except Exception as e:
        print(f"Error during vLLM generation for request_id {request_id}: {e}")
        if websocket.client_state == WebSocketState.CONNECTED: 
            await websocket.send_text(f"[ERROR] LLM generation failed: {e}")
        if chat_history and chat_history[-1]["role"] == "user":
            chat_history.pop() 
        return 

    if websocket.client_state == WebSocketState.CONNECTED: 
        await websocket.send_text(f"[LLM_RESPONSE_TEXT] {full_llm_response_text}")
    cprint(f"LLM full response: {full_llm_response_text}", 'blue')

    chat_history.append({"role": "assistant", "content": full_llm_response_text})

    if tts_stream and tts_engine and full_llm_response_text: 
        print(f"TTS: Synthesizing and streaming audio for: \"{full_llm_response_text[:100]}...\"")
        
        loop = asyncio.get_event_loop()
        async def send_tts_chunk(chunk_bytes: bytes):
            if websocket.client_state == WebSocketState.CONNECTED:
                await websocket.send_bytes(chunk_bytes)
            else:
                print("TTS Callback: WebSocket no longer connected. Cannot send audio chunk.")

        def sync_on_tts_chunk_callback(audio_chunk_bytes: bytes):
            try:
                if not loop.is_closed():
                    asyncio.run_coroutine_threadsafe(send_tts_chunk(audio_chunk_bytes), loop)
                else:
                    print("TTS Callback: Event loop is closed. Cannot send audio chunk.")
            except Exception as e:
                print(f"Error scheduling/sending TTS audio chunk: {e}")

        try:

            #FIXME
            text_generator = lambda: (yield full_llm_response_text)
            tts_stream.feed(text_generator())
            print("-----------------")
            cprint(full_llm_response_text)
            print(full_llm_response_text)
            await asyncio.to_thread(
                tts_stream.play,
                on_audio_chunk=sync_on_tts_chunk_callback
            )
            if websocket.client_state == WebSocketState.CONNECTED:
                 await websocket.send_text("[TTS_DONE]")
            else:
                 print("TTS: Websocket not connected, cannot send TTS_DONE.")
            print("TTS: Audio streaming complete.")

        except Exception as e:
            print(f"Error during TTS synthesis or streaming: {e}")
            if websocket.client_state == WebSocketState.CONNECTED:
                await websocket.send_text(f"[ERROR] TTS failed: {str(e)}") 
    elif not (tts_stream and tts_engine): 
        print("[INFO] TTS engine not available. Cannot generate audio.") 
        if websocket.client_state == WebSocketState.CONNECTED:
            await websocket.send_text("[INFO] TTS engine not available. Cannot generate audio.") 
            await websocket.send_text("[TTS_DONE]") 
    elif not full_llm_response_text:
        if websocket.client_state == WebSocketState.CONNECTED:
            await websocket.send_text("[TTS_DONE]")


if __name__ == "__main__":
    import uvicorn
    if not os.path.exists(TTS_REFERENCE_VOICE_WAV):
        print(f"Warning: TTS Reference voice file '{TTS_REFERENCE_VOICE_WAV}' not found. A default voice will be used.")
    
    # Ensure uvicorn uses the 'websockets' library for WebSocket handling
    uvicorn.run(app, host="0.0.0.0", port=8000, ws="websockets")