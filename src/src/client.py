import asyncio
import websockets
# Try importing ConnectionState from different locations based on library version
try:
    # websockets version 10.x+
    from websockets.connection import ConnectionState
except ImportError:
    try:
        # websockets version 8.x, 9.x
        from websockets.protocol import ConnectionState
    except ImportError:
        try:
            # Another possible location for older versions or specific forks
            from websockets.legacy.protocol import ConnectionState
        except ImportError:
            # Fallback if ConnectionState cannot be imported
            print("Warning: Could not import ConnectionState from common locations "
                  "(websockets.connection, websockets.protocol, websockets.legacy.protocol). "
                  "Connection checks will rely on exceptions or basic attribute checks if available.")
            ConnectionState = None # type: ignore

import soundfile as sf
import os
import numpy as np 
import time 
import wave

# --- Configuration ---
VAD_SAMPLE_RATE = 16000
MIN_SILENCE_DURATION_MS = 700
SPEECH_PAD_MS = 200
TTS_AUDIO_OUTPUT_DIR = "../../outputs/received_tts_audio"
COMBINED_TTS_AUDIO_FILENAME = "full_received_response.wav"
TTS_CHUNK_PREFIX = "received_chunk_"

# Expected format from server's TTS
TTS_SERVER_SAMPLE_RATE = 24000
TTS_SERVER_CHANNELS = 1
TTS_SERVER_SAMPLE_WIDTH_BYTES = 2 # 2 bytes for int16 (PCM)

# --- Global state for current turn's audio ---
current_turn_audio_data_list = []
current_turn_chunk_counter = 0

def ensure_output_dir():
    """Creates the output directory for received audio if it doesn't exist."""
    os.makedirs(TTS_AUDIO_OUTPUT_DIR, exist_ok=True)

def save_audio_chunk(audio_bytes: bytes, turn_chunk_idx: int):
    """Saves a single audio chunk (received as bytes) to a WAV file."""
    global current_turn_audio_data_list
    try:
        output_filename = os.path.join(TTS_AUDIO_OUTPUT_DIR, f"{TTS_CHUNK_PREFIX}{turn_chunk_idx:04d}.wav")
        
        audio_data_int16 = np.frombuffer(audio_bytes, dtype=np.int16)

        with wave.open(output_filename, 'wb') as wf:
            wf.setnchannels(TTS_SERVER_CHANNELS)
            wf.setsampwidth(TTS_SERVER_SAMPLE_WIDTH_BYTES)
            wf.setframerate(TTS_SERVER_SAMPLE_RATE)
            wf.writeframes(audio_data_int16.tobytes()) 
        
        current_turn_audio_data_list.append(audio_data_int16)

    except Exception as e:
        print(f"Error saving audio chunk {turn_chunk_idx}: {e}")

def save_combined_audio(audio_data_list: list, output_filename_base: str):
    """Combines and saves all audio chunks for the current turn."""
    if not audio_data_list:
        print("No audio data to combine for this turn.")
        return

    combined_filepath = os.path.join(TTS_AUDIO_OUTPUT_DIR, output_filename_base)

    try:
        combined_data = np.concatenate(audio_data_list)
        with wave.open(combined_filepath, 'wb') as wf:
            wf.setnchannels(TTS_SERVER_CHANNELS)
            wf.setsampwidth(TTS_SERVER_SAMPLE_WIDTH_BYTES)
            wf.setframerate(TTS_SERVER_SAMPLE_RATE)
            wf.writeframes(combined_data.tobytes())
        print(f"\nSuccessfully saved combined TTS audio to: {combined_filepath}")
    except Exception as e:
        print(f"Error saving combined TTS audio: {e}")

# Function to read audio file bytes (for sending to server)
def get_audio_bytes_to_send(audio_file_path: str):
    if not os.path.exists(audio_file_path):
        print(f"Error: Audio file '{audio_file_path}' not found.")
        return None
    
    data, samplerate = sf.read(audio_file_path, dtype='int16') # Server VAD expects 16kHz int16
    
    if samplerate != VAD_SAMPLE_RATE:
        print(f"Warning: Audio file '{audio_file_path}' is {samplerate}Hz. Converting/resampling might be needed if server strictly expects {VAD_SAMPLE_RATE}Hz.")
        # For simplicity, this example does not resample. Ensure input audio matches VAD_SAMPLE_RATE.
    
    if data.ndim > 1: # Ensure mono
        data = data[:, 0] 
    
    return data.tobytes()

def generate_silence_bytes(duration_ms: int, sample_rate: int = VAD_SAMPLE_RATE) -> bytes:
    """Generates a byte array of silence (zeros) for a given duration at specified sample rate."""
    num_samples = int(duration_ms * sample_rate / 1000)
    silence_np = np.zeros(num_samples, dtype=np.int16)
    return silence_np.tobytes()

def is_websocket_really_open(websocket_protocol: websockets.WebSocketClientProtocol):
    """
    Checks if the WebSocket is open using available attributes.
    Relies on ConnectionState if available, otherwise tries .closed attribute.
    """
    if ConnectionState: # Check if ConnectionState was successfully imported
        return websocket_protocol.state == ConnectionState.OPEN
    else:
        # Fallback if ConnectionState couldn't be imported
        # The .closed attribute is generally more reliable than .open for older versions.
        # If 'closed' is True, it's definitely closed. If False, it's likely open or connecting.
        # If the attribute doesn't exist, we assume open and let operations fail.
        try:
            # For websockets library, the 'closed' attribute indicates if the closing handshake is done.
            # 'open' attribute is also common.
            if hasattr(websocket_protocol, 'open') and websocket_protocol.open:
                return True
            if hasattr(websocket_protocol, 'closed') and not websocket_protocol.closed:
                return True
            # If neither specific attribute helps, we assume it's open if no exception during operations.
            # This function becomes more of a best-effort check if ConnectionState is not available.
            # The primary check will be the try-except blocks around send/recv.
            return True # Optimistic assumption if specific attributes are not definitive or present
        except AttributeError:
            # If .open or .closed attribute is not found, we make an optimistic assumption.
            print("is_websocket_really_open: Cannot reliably determine state without ConnectionState or standard attributes. Assuming open.")
            return True


async def send_and_receive_turn(websocket: websockets.WebSocketClientProtocol, 
                                prompt: str = None, 
                                audio_file_path: str = None, 
                                post_audio_silence_ms: int = 0):
    """
    Sends either a text prompt or an audio file over WebSocket.
    Then, listens for and processes text responses, LLM text, and TTS audio chunks.
    """
    global current_turn_audio_data_list, current_turn_chunk_counter
    
    # Reset for the new turn
    current_turn_audio_data_list = []
    current_turn_chunk_counter = 0
    # ensure_output_dir() # Called once at the start of main now

    if audio_file_path:
        audio_bytes_to_send = get_audio_bytes_to_send(audio_file_path)
        if audio_bytes_to_send is None:
            return False 
        
        full_audio_stream_bytes = audio_bytes_to_send
        if post_audio_silence_ms > 0:
            silence_bytes = generate_silence_bytes(post_audio_silence_ms, sample_rate=VAD_SAMPLE_RATE)
            full_audio_stream_bytes += silence_bytes
            print(f"\nStreaming {len(audio_bytes_to_send)} audio bytes from '{audio_file_path}' + {post_audio_silence_ms}ms silence...")
        else:
            print(f"\nStreaming {len(audio_bytes_to_send)} audio bytes from '{audio_file_path}'...")

        # Simulate streaming audio by sending in chunks
        CHUNK_SIZE_BYTES = 3200 # 100ms at 16KHz, 16-bit mono (16000 * 2 * 0.1)
        for i in range(0, len(full_audio_stream_bytes), CHUNK_SIZE_BYTES):
            if not is_websocket_really_open(websocket): # Check before sending
                print("WebSocket connection closed before sending all audio chunks.")
                return False
            try:
                chunk = full_audio_stream_bytes[i:i + CHUNK_SIZE_BYTES]
                await websocket.send(chunk)
                await asyncio.sleep(0.09) # Slightly less than chunk duration to simulate real-time
            except websockets.exceptions.ConnectionClosed:
                print("WebSocket connection closed during audio send.")
                return False
        
        # Wait for server VAD to process the tail end of speech/silence
        await asyncio.sleep((MIN_SILENCE_DURATION_MS + SPEECH_PAD_MS) / 1000 + 0.3) # Adjusted wait
        
    elif prompt:
        print(f"\nSending text prompt: '{prompt}'")
        if not is_websocket_really_open(websocket): # Check before sending
            print("WebSocket connection closed before sending text prompt.")
            return False
        try:
            await websocket.send(prompt)
        except websockets.exceptions.ConnectionClosed:
            print("WebSocket connection closed during text send.")
            return False
    else:
        print("Error: No prompt or audio file provided for this turn.")
        return False

    # --- Receiving Loop ---
    print("Waiting for server response...")
    llm_full_text_response = ""
    receiving_tts = False

    try:
        while True:
            if not is_websocket_really_open(websocket):
                print("WebSocket connection closed while waiting for/receiving server response.")
                if receiving_tts and current_turn_audio_data_list: 
                    save_combined_audio(current_turn_audio_data_list, COMBINED_TTS_AUDIO_FILENAME + "_incomplete_closed_early")
                return False 
            
            try:
                message = await asyncio.wait_for(websocket.recv(), timeout=60.0) 
            except asyncio.TimeoutError:
                print("\nTimeout waiting for message from server.")
                if receiving_tts and current_turn_audio_data_list: 
                    save_combined_audio(current_turn_audio_data_list, COMBINED_TTS_AUDIO_FILENAME + "_incomplete_timeout")
                return False 
            except websockets.exceptions.ConnectionClosed as e:
                print(f"\nWebSocket connection closed by server while waiting for message: {e}")
                if receiving_tts and current_turn_audio_data_list:
                    save_combined_audio(current_turn_audio_data_list, COMBINED_TTS_AUDIO_FILENAME + "_incomplete_closed_recv")
                return False


            if isinstance(message, bytes): # This is an audio chunk from TTS
                if not receiving_tts: # First audio chunk for this turn
                    print("\n[TTS_STREAM_START] Receiving TTS audio chunks...")
                    receiving_tts = True
                
                current_turn_chunk_counter += 1
                save_audio_chunk(message, current_turn_chunk_counter)
                print(f" .", end="", flush=True) # Indicate audio chunk received

            elif isinstance(message, str):
                if message.startswith("[LLM_RESPONSE_TEXT]"):
                    llm_full_text_response = message[len("[LLM_RESPONSE_TEXT] "):]
                    print(f"\n\n[LLM SAID]:\n{llm_full_text_response}")
                elif message == "[TTS_DONE]":
                    print("\n[TTS_STREAM_END] Server signaled end of TTS audio.")
                    if current_turn_audio_data_list: 
                        save_combined_audio(current_turn_audio_data_list, COMBINED_TTS_AUDIO_FILENAME)
                    else:
                        print("No TTS audio chunks were received for this response.")
                    current_turn_audio_data_list = [] 
                    current_turn_chunk_counter = 0
                    receiving_tts = False
                    return True 
                elif message.startswith("[ERROR]"):
                    print(f"\n[SERVER_ERROR]: {message[len('[ERROR] '):]}")
                    return False 
                elif message.startswith(("[INFO]", "[WARNING]", "[VOICE_START]", "[VOICE_END]", "[TRANSCRIPTION]", "[PROMPT]")):
                    if message.startswith("[PROMPT]"):
                        print(f"\n[SERVER_ACK_PROMPT]: {message[len('[PROMPT] '):]}")
                    elif message.startswith("[TRANSCRIPTION]"):
                        print(f"\n[SERVER_TRANSCRIPTION]: {message[len('[TRANSCRIPTION] '):]}")
                    elif message == "[VOICE_START]":
                        print("\n[VAD_SERVER]: Voice activity started.")
                    elif message == "[VOICE_END]":
                        print("\n[VAD_SERVER]: Voice activity ended by server.")
                    else: 
                        print(f"\n[SERVER_MESSAGE]: {message}")
                else: 
                    print(f"\n[UNHANDLED_SERVER_TEXT]: {message}")
            else:
                print(f"\n[UNKNOWN_MESSAGE_TYPE]: {type(message)}")

    except websockets.exceptions.ConnectionClosed as e:
        print(f"\nWebSocket connection closed during turn (outer try): {e}")
        if receiving_tts and current_turn_audio_data_list: 
            save_combined_audio(current_turn_audio_data_list, COMBINED_TTS_AUDIO_FILENAME + "_incomplete_closed_outer")
        return False 
    except Exception as e:
        print(f"\nAn unexpected error occurred during message reception (outer try): {e}")
        if receiving_tts and current_turn_audio_data_list: 
            save_combined_audio(current_turn_audio_data_list, COMBINED_TTS_AUDIO_FILENAME + "_incomplete_error_outer")
        return False 
    
    print("Warning: send_and_receive_turn exited receive loop unexpectedly.")
    if receiving_tts and current_turn_audio_data_list:
        print("\nExited unexpectedly during TTS. Saving any received audio.")
        save_combined_audio(current_turn_audio_data_list, COMBINED_TTS_AUDIO_FILENAME + "_incomplete_unexpected_exit")
    return False


async def main():
    uri = "ws://localhost:8000/ws"
    print(f"Attempting to connect to WebSocket server at {uri}...")
    ensure_output_dir() 

    test_audio_file = "voice.wav"
    if not os.path.exists(test_audio_file):
        print(f"Test audio file '{test_audio_file}' not found. Creating a dummy one.")
        dummy_silence = np.zeros(16000, dtype=np.int16) 
        sf.write(test_audio_file, dummy_silence, 16000, subtype='PCM_16')
        print(f"Created dummy '{test_audio_file}'.")


    try:
        async with websockets.connect(uri, ping_interval=20, ping_timeout=20) as websocket:
            print("Connected to WebSocket server.")

            interactions = [
                {"id": 1, "type": "text", "prompt": "Hello assistant! Can you tell me a very short story about a brave robot?"},
                # {"id": 2, "type": "audio", "audio_file_path": test_audio_file, "post_audio_silence_ms": 1000},
                {"id": 3, "type": "text", "prompt": "That was interesting. What is your name?"}
            ]

            for interaction_config in interactions:
                interaction_id = interaction_config["id"]
                interaction_type = interaction_config["type"]
                print(f"\n" + "="*30 + f" Interaction {interaction_id}: {interaction_type.capitalize()} Input " + "="*30)
                
                if not is_websocket_really_open(websocket): 
                    print(f"Connection lost before starting Interaction {interaction_id}.")
                    return 
                
                success = False
                if interaction_type == "text":
                    success = await send_and_receive_turn(websocket, prompt=interaction_config["prompt"])
                elif interaction_type == "audio":
                    success = await send_and_receive_turn(websocket, 
                                                          audio_file_path=interaction_config["audio_file_path"], 
                                                          post_audio_silence_ms=interaction_config["post_audio_silence_ms"])
                
                if not is_websocket_really_open(websocket): 
                    print(f"Connection lost after Interaction {interaction_id}.")
                    return 
                
                if not success:
                    print(f"Interaction {interaction_id} failed or did not complete successfully.")
                    # If a turn fails and the connection is still open, we might want to stop.
                    # However, the current logic will attempt the next interaction if the connection is still open.
                    # If the server error caused a disconnect, the `is_websocket_really_open` check
                    # at the start of the next iteration should catch it.

                if interaction_id < len(interactions): 
                    await asyncio.sleep(1) 
            
            print("\nAll specified interactions attempted.")

    except ConnectionRefusedError:
        print(f"\nConnection refused. Is the server running at {uri}?")
    except websockets.exceptions.InvalidStatusCode as e: 
        print(f"\nConnection failed with status code: {e.status_code}. Server might be misconfigured or down.")
    except asyncio.TimeoutError: 
        print(f"\nConnection attempt to {uri} timed out.")
    except Exception as e:
        print(f"\nAn unexpected error occurred in main: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())