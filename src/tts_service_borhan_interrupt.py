import asyncio
import functools  # For functools.partial
import logging  # For better logging control
import os
import queue
import threading
from pathlib import Path

import numpy as np
import torch
import websockets
import yaml
from fish_engine import (
    FishEngine,  # Assuming FishEngine is the TTS engine you want to use
)

# Configure for headless server environment
# These might not be strictly necessary for a server that only generates audio,
# but are good practice if PyAudio or other audio libraries are used for playback/capture.
os.environ["ALSA_PCM_DEVICE"] = "0"
os.environ["ALSA_PCM_CARD"] = "0"
os.environ["SDL_AUDIODRIVER"] = "dummy"
# os.environ['PULSE_RUNTIME_PATH'] = '/tmp/pulse' # Uncomment if you use PulseAudio and need a specific runtime path

# RealtimeTTS imports
from RealtimeTTS import TextToAudioStream

# Load YAML configuration first so it can be used in route definitions
parent_path = Path(__file__).parents[1]
yaml_path = Path(__file__).parents[2] / "config.yaml"
print(yaml_path)
yaml_config = yaml.safe_load(open(yaml_path, "r"))

# Configure logging for the server
# Create logs directory if it doesn't exist
os.makedirs(os.path.join(parent_path, "outputs"), exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    filename=os.path.join(parent_path, "outputs", "tts_server.log"),
)
logger = logging.getLogger(__name__)


# --- Configuration ---
SERVER_HOST = yaml_config["tts"]["host"]  # Allow connections from outside the container
SERVER_PORT = yaml_config["tts"]["port"]  # Match the exposed port in Dockerfile 8765

# Audio configuration
COQUI_SAMPLE_RATE = 24000  # CoquiEngine outputs at 24kHz
TARGET_SAMPLE_RATE = 8000
TARGET_SAMPLE_WIDTH_BYTES = 2  # 16-bit PCM

# Manual buffer configuration for increased fluency
BUFFER_SEND_THRESHOLD_MS = (
    500  # Send audio when buffer accumulates this many milliseconds (e.g., 150-200ms)
)
# Calculate buffer size in bytes, ensuring it's a multiple of the sample width
BUFFER_SEND_THRESHOLD_BYTES = int(
    TARGET_SAMPLE_RATE * TARGET_SAMPLE_WIDTH_BYTES * (BUFFER_SEND_THRESHOLD_MS / 1000.0)
)
BUFFER_SEND_THRESHOLD_BYTES = (
    BUFFER_SEND_THRESHOLD_BYTES // TARGET_SAMPLE_WIDTH_BYTES
) * TARGET_SAMPLE_WIDTH_BYTES

# Path to your reference voice WAV file for cloning.
REFERENCE_VOICE_WAV = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "egyption_voice.wav"
)  # Use absolute path
voice_counter = 0

# Global CoquiEngine instance
global_coqui_engine = None


# Simple, fast audio resampling without buffering delays
def resample_audio_chunk(
    audio_bytes, source_rate=COQUI_SAMPLE_RATE, target_rate=TARGET_SAMPLE_RATE
):
    """Simple, fast resampling for real-time audio"""
    try:
        if len(audio_bytes) == 0:
            return b""

        # Convert bytes to numpy array (assuming 16-bit PCM)
        audio_int16 = np.frombuffer(audio_bytes, dtype=np.int16)

        if len(audio_int16) == 0:
            return b""

        # Simple decimation for 24kHz -> 8kHz (factor of 3)
        # This is fast and works well for this specific ratio
        if source_rate % target_rate == 0:
            decimated = audio_int16[:: source_rate // target_rate]
        else:
            # Fallback: convert to float, use simple linear interpolation
            audio_float = audio_int16.astype(np.float32) / 32768.0

            # Calculate new length
            new_length = int(len(audio_float) * target_rate / source_rate)
            if new_length == 0:
                return b""

            # Simple linear interpolation resampling
            old_indices = np.linspace(0, len(audio_float) - 1, len(audio_float))
            new_indices = np.linspace(0, len(audio_float) - 1, new_length)
            decimated_float = np.interp(new_indices, old_indices, audio_float)
            decimated = (decimated_float * 32767).astype(np.int16)

        return decimated.tobytes()

    except Exception as e:
        logger.error(f"Error resampling audio: {e}")
        return b""  # Return empty bytes on error to prevent sending bad data


# --- TTS Generator Wrapper (now takes a queue) ---
def tts_input_generator(client_word_queue):
    """
    This generator yields words received from a specific client via its word_queue.
    It blocks until a word is available or a None sentinel is received.
    """
    while True:
        word = client_word_queue.get()  # Blocks until a word is available
        print(f"Received word: {word}")  # Debug print
        if word is None:  # Sentinel to stop the generator
            logger.info("TTS input generator received stop signal.")
            print("TTS input generator received stop signal.")
            break
        yield word


# --- Warm-up Function ---
def warm_up_engine(engine_instance):
    """
    Performs a dummy synthesis to warm up the TTS engine.
    """
    logger.info("Warming up TTS engine...")
    print("Warming up TTS engine...")
    # Use a temporary TextToAudioStream for the warm-up
    # It needs a dummy callback, but we don't care about the output
    dummy_stream = TextToAudioStream(engine=engine_instance, muted=True)
    dummy_text = "Hello."  # Short, simple text
    dummy_stream.feed(dummy_text)

    # Use a simple lambda for on_audio_chunk as we don't care about the output
    # The important part is that the synthesis process runs.
    dummy_stream.play(
        on_audio_chunk=lambda x: None,
        log_synthesized_text=False,
        fast_sentence_fragment=True,  # Make warm-up fast
    )
    dummy_stream.stop()  # Clean up the dummy stream
    logger.info("TTS engine warmed up.")
    print("TTS engine warmed up.")


# --- Manual Audio Buffer Sender Task ---
async def send_buffered_audio_task(
    websocket,
    client_address,
    audio_buffer,
    buffer_lock,
    buffer_event,
    done_generating_audio_event,
):
    """
    This task sends accumulated audio chunks from the buffer over the WebSocket.
    It waits for a signal that new audio is available or for a flush signal.
    """
    logger.info(f"Starting buffered audio sender for client {client_address}")

    # Small timeout for periodic checks, even if no event is set, to ensure timely flushes
    # when done_generating_audio_event is set but buffer_event isn't (e.g., very small last chunk)
    PERIODIC_CHECK_INTERVAL = 0.05  # seconds

    try:
        while True:
            # Wait for a signal or a timeout for periodic check
            try:
                await asyncio.wait_for(
                    buffer_event.wait(), timeout=PERIODIC_CHECK_INTERVAL
                )
                buffer_event.clear()  # Clear the event after it's set
            except asyncio.TimeoutError:
                pass  # Timeout occurred, proceed to check buffer anyway

            # Loop to send chunks as long as there's enough data or we're flushing
            while True:
                chunk_to_send = b""
                with buffer_lock:
                    if len(audio_buffer) >= BUFFER_SEND_THRESHOLD_BYTES:
                        # Extract a chunk of the threshold size
                        chunk_to_send = bytes(
                            audio_buffer[:BUFFER_SEND_THRESHOLD_BYTES]
                        )
                        del audio_buffer[:BUFFER_SEND_THRESHOLD_BYTES]
                    elif done_generating_audio_event.is_set() and len(audio_buffer) > 0:
                        # If TTS is done and there's remaining audio, send it all
                        chunk_to_send = bytes(audio_buffer)
                        audio_buffer.clear()
                    else:
                        # Not enough data to send a full chunk, and not flushing yet
                        break

                if chunk_to_send:
                    try:
                        await websocket.send(chunk_to_send)
                        # logger.debug(f"Sent {len(chunk_to_send)} bytes to {client_address}")
                    except websockets.exceptions.ConnectionClosed:
                        logger.warning(
                            f"WebSocket closed for {client_address} while sending buffered audio. Exiting sender task."
                        )
                        return  # Exit task if connection is gone
                    except Exception as e:
                        logger.error(
                            f"Error sending buffered audio to {client_address}: {e}"
                        )
                        return  # Exit task on send error
                else:
                    # No chunk was prepared in this iteration, break from inner loop
                    break

            # If TTS is done and the buffer is empty, we can exit this task
            if done_generating_audio_event.is_set() and len(audio_buffer) == 0:
                logger.info(
                    f"Buffered audio sender for {client_address}: All audio flushed, exiting."
                )
                break

    except asyncio.CancelledError:
        logger.info(f"Buffered audio sender for {client_address} cancelled.")
    except Exception as e:
        logger.exception(
            f"Unexpected error in buffered audio sender for {client_address}: {e}"
        )
    finally:
        logger.info(f"Buffered audio sender for {client_address} finished.")


# --- WebSocket Handler ---
async def handle_client(websocket):
    client_address = websocket.remote_address
    logger.info(f"Client connected from {client_address}")
    print(f"Client connected from {client_address}")

    client_word_queue = queue.Queue()
    current_loop = asyncio.get_running_loop()

    # --- Manual Buffer Setup ---
    audio_buffer = bytearray()
    buffer_lock = threading.Lock()  # Protects audio_buffer from concurrent access
    buffer_event = asyncio.Event()  # Signals the sender task that new data is available
    done_generating_audio_event = (
        asyncio.Event()
    )  # Signals that TTS has finished generating all audio

    # Initialize TextToAudioStream for this client, using the global warmed-up engine
    try:
        stream = TextToAudioStream(
            engine=global_coqui_engine,
            muted=True,
            log_characters=False,
        )
    except Exception as e:
        logger.error(
            f"Error initializing TextToAudioStream for client {client_address}: {e}"
        )
        await websocket.close()
        return

    # --- Audio Chunk Callback (runs in RealtimeTTS's internal thread) ---
    def on_audio_chunk_callback(audio_chunk_bytes):
        try:
            if len(audio_chunk_bytes) == 0:
                return

            # Resample immediately before adding to buffer
            resampled_audio = resample_audio_chunk(audio_chunk_bytes)

            if len(resampled_audio) > 0:
                with buffer_lock:
                    audio_buffer.extend(resampled_audio)
                buffer_event.set()  # Signal the sender task that new data is available

        except Exception as e:
            logger.error(
                f"Error processing audio chunk for client {client_address}: {e}"
            )
            # In a production system, you might want to signal a client disconnect here
            # or stop the TTS stream if this error is critical.

    global voice_counter
    voice_counter += 1
    # --- Define the blocking TTS play function to be run in executor ---
    play_kwargs = {
        "on_audio_chunk": on_audio_chunk_callback,
        "log_synthesized_text": True,
        "buffer_threshold_seconds": 0.2,  # RealtimeTTS's internal buffer, can be small
        "output_wavfile": f"generated_audio_1.wav",
    }

    logger.info(
        f"Starting TTS synthesis for client {client_address} with settings: {play_kwargs}"
    )
    play_callable = functools.partial(stream.play, **play_kwargs)

    # --- Task to run TTS synthesis in a separate thread ---
    tts_play_task = current_loop.run_in_executor(None, play_callable)

    # --- Task to receive messages from the client ---
    async def receive_client_messages():
        try:
            stream.feed(
                tts_input_generator(client_word_queue)
            )  # Feed the generator to the stream
            async for message in websocket:
                if message == "END_OF_TEXT":
                    logger.info(
                        f"Received END_OF_TEXT from client {client_address}. Signaling TTS generator to stop."
                    )
                    print(
                        f"Received END_OF_TEXT from client {client_address}. Signaling TTS generator to stop."
                    )
                    client_word_queue.put(None)  # Signal the generator to stop
                    break  # Stop receiving messages
                logger.info(f"Server received token from {client_address}: '{message}'")
                client_word_queue.put(message)
        except websockets.exceptions.ConnectionClosedOK:
            logger.info(
                f"Client {client_address} disconnected gracefully during message reception."
            )
        except websockets.exceptions.ConnectionClosedError as e:
            logger.warning(
                f"Client {client_address} disconnected with error during message reception: {e}"
            )
        except Exception as e:
            logger.exception(
                f"Error in receive_client_messages for client {client_address}: {e}"
            )
        finally:
            # Ensure the queue is signaled to stop if the message reception task ends prematurely
            client_word_queue.put(None)
            logger.info(f"Receive messages task for {client_address} finished.")

    receive_messages_task = asyncio.create_task(receive_client_messages())
    send_audio_task = asyncio.create_task(
        send_buffered_audio_task(
            websocket,
            client_address,
            audio_buffer,
            buffer_lock,
            buffer_event,
            done_generating_audio_event,
        )
    )

    # --- Wait for tasks to complete and handle cleanup ---
    try:
        # Wait for message reception or TTS generation to complete first
        # This allows us to react when the client stops sending text or TTS finishes
        done, pending = await asyncio.wait(
            [receive_messages_task, tts_play_task], return_when=asyncio.FIRST_COMPLETED
        )

        # Ensure TTS generator is signaled to stop if not already (e.g., if client disconnected)
        client_word_queue.put(None)

        # If TTS play task is still pending, wait for it to truly finish.
        # This ensures all audio chunks are generated and passed to on_audio_chunk_callback.
        if tts_play_task in pending:
            logger.info(
                f"Waiting for TTS play task to complete for {client_address}..."
            )
            await tts_play_task

        # Signal the audio sender that no more audio will be generated
        done_generating_audio_event.set()
        buffer_event.set()  # Wake up the sender one last time to flush remaining audio

        # Wait for the audio sender task to complete its flush and exit
        logger.info(
            f"Waiting for audio sender task to complete for {client_address}..."
        )
        await send_audio_task  # This will wait until the sender flushes and exits

        # Ensure any remaining pending tasks are cancelled (should mostly be receive_messages_task if it wasn't done)
        for task in pending:
            if (
                task != tts_play_task and task != send_audio_task
            ):  # Don't cancel tasks we just awaited
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass  # Task was cancelled as expected

        # Check for exceptions in the completed tasks
        for task in done:
            if task.exception():
                logger.error(f"Task completed with exception: {task.exception()}")
        if tts_play_task.exception():
            logger.error(
                f"TTS play task completed with exception: {tts_play_task.exception()}"
            )
        if send_audio_task.exception():
            logger.error(
                f"Audio send task completed with exception: {send_audio_task.exception()}"
            )

        # Signal client that all audio has been sent
        await websocket.send("END_OF_AUDIO")
        logger.info(f"Sent END_OF_AUDIO signal to client {client_address}.")

        # --- Final Handshake: Wait for client ACK ---
        try:
            ack_message = await asyncio.wait_for(
                websocket.recv(), timeout=10
            )  # Wait for ACK with timeout
            if ack_message == "ACK_AUDIO_RECEIVED":
                logger.info(
                    f"Received ACK_AUDIO_RECEIVED from client {client_address}. Handshake complete."
                )
            else:
                logger.warning(
                    f"Received unexpected final message from client {client_address}: '{ack_message}'"
                )
        except asyncio.TimeoutError:
            logger.warning(
                f"Client {client_address} did not send ACK_AUDIO_RECEIVED within timeout."
            )
        except websockets.exceptions.ConnectionClosedOK:
            logger.info(
                f"Client {client_address} closed connection before sending ACK."
            )
        except websockets.exceptions.ConnectionClosedError as e:
            logger.warning(
                f"Client {client_address} connection closed with error during ACK wait: {e}"
            )

    except websockets.exceptions.ConnectionClosedOK:
        logger.info(f"Client {client_address} disconnected gracefully.")
    except websockets.exceptions.ConnectionClosedError as e:
        logger.warning(f"Client {client_address} disconnected with error: {e}")
    except Exception as e:
        logger.exception(
            f"An unexpected error occurred for client {client_address}: {e}"
        )
    finally:
        # Ensure TTS stream is stopped and engine resources are released for this stream
        logger.info(f"Cleaning up TTS stream resources for client {client_address}...")
        stream.stop()  # This also signals the TTS generator to stop if not already

        # Ensure all tasks are cancelled if they are still running
        for task in [receive_messages_task, tts_play_task, send_audio_task]:
            if not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

        logger.info(f"Client {client_address} disconnected.")


async def main():
    global global_coqui_engine

    # Create a dummy reference voice file if it doesn't exist, for initial testing
    if not os.path.exists(REFERENCE_VOICE_WAV):
        try:
            import soundfile as sf

            # Create 1 second of silence at 22050 Hz, 16-bit PCM
            sf.write(REFERENCE_VOICE_WAV, np.zeros(22050), 22050, subtype="PCM_16")
            logger.info(f"Created dummy reference voice file: {REFERENCE_VOICE_WAV}")
        except ImportError:
            logger.warning(
                f"'{REFERENCE_VOICE_WAV}' not found and soundfile not installed to create dummy. Coqui will use a default voice."
            )
        except Exception as e:
            logger.warning(
                f"Could not create dummy reference voice file: {e}. Coqui will use a default voice."
            )

    # Initialize the global CoquiEngine instance once
    logger.info("Initializing global CoquiEngine...")
    print("\033[93mInitializing global CoquiEngine...\033[0m")
    try:
        # prompt = """مرحباً يا جماعة، كيفكم؟ اليوم حاب أحكي عن شغلة بنمرّ فيها كلنا، يمكن ما بنعطيها اهتمام، بس إلها تأثير كبير علينا,الضغط اليومي. يعني من أول ما نصحى، وإحنا برتم سريع, شغل، جامعة، مشاوير، التزامات، والزلمة ما بلحق حتى يشرب قهوته ع رواق. كل يوم نفس الروتين، وكأننا داخلين سباق، وكل ما نقول خلص، هاليوم بمرّ، بنلاقي حالنا بدوامة تانية! طب وإمتى آخر مرّة فعلياً قعدت مع حالك؟ لا تليفون، لا تليفونات، لا مشاغل,بس لحظة هدوء؟ صرنا ننسى نعيش اللحظة، نضحك من قلبنا، نستمتع بشغلة بسيطة زي كاسة شاي، أو قعدة عالبلكونة وقت المغرب. أنا بقول, لازم نروّق شوي، نخفف السرعة، ونفكر بنفسنا شوي، لأنو صحتك النفسية مش إشي ثانوي، هاد أهم إشي. خدلك لحظة كل يوم، احكي مع حالك، رتب أفكارك، وافهم شو بتحتاج,الدنيا مش مستاهلة كل هالركض، وإذا ما ريّحت حالك، محدا رح يجي يريحك.وبس، هاي كانت دقيقتي معكم اليوم، إذا حبيتوا الموضوع، احكولي, وديروا بالكم على حالكم، والله يعطيكم العافية."""
        # prompt = """Everything you do is a standing ovation on your first record if you're having that breakthrough record. And then you put out your second body of work and then you realize that everything you're putting out now is being compared to what they liked about your first record. But then you put out the third one and then it's compared to the first two. Then you put out the fourth one. Then it's compared to the first three and it goes on and on and on. By the time you're at album seven, you are so, you have such a strange, convoluted relationship with your previous work because you're like, damn it, All Too Well was a good song. And I knew with this album, it was like something that was almost a return to form. Like Reputation was such an important record for me because I couldn't stop writing. And I needed to write that album and I needed to put out that album and I needed to not explain that album. Because another thing about that album was I knew if I did an interview about it, none of it would be about music. And this entire lover phase"""
        prompt = """topic but I think honestly I didn't there was no thought process to this album for me really it was all just I mean thought process in the sense of I'm going through something and trying to figure out where I'm at and what I'm feeling and what I'm going to do that but like as far as like oh this is a song that I'm going to put on a record like that wasn't it um it was really just a lot of it like before you got here I was listening to with your people and I was like, man, this is like, I was very angry. Yeah. And foundation being hurt. So yeah, so it's really just, yeah, I didn't really think about it. And I do like the idea of taking a quirky pop happy sound melodically and like the sound of the"""
        global_coqui_engine = FishEngine(
            checkpoint_dir=os.path.join(
                parent_path, "checkpoints", "openaudio-s1-mini"
            ),
            output_dir=os.path.join(parent_path, "outputs"),
            voices_path=None,
            reference_prompt_tokens=os.path.join(parent_path, "ref_kelly.npy"),
            device="cuda",
            precision=torch.bfloat16,  # "half",
            reference_prompt_text=prompt,
        )
        logger.info("Global FishEngine initialized.")
        print("\033[93mGlobal FishEngine initialized.\033[0m")

        # Warm up the engine
        warm_up_engine(global_coqui_engine)

    except Exception as e:
        logger.critical(f"Fatal error initializing global CoquiEngine: {e}")
        logger.critical("Server cannot start without a working TTS engine.")
        return

    logger.info(f"Starting WebSocket server on ws://{SERVER_HOST}:{SERVER_PORT}")
    print(f"Starting WebSocket server on ws://{SERVER_HOST}:{SERVER_PORT}")
    logger.info(
        f"Audio processing: {COQUI_SAMPLE_RATE}Hz -> {TARGET_SAMPLE_RATE}Hz with fast decimation"
    )
    logger.info(
        f"Manual buffer: {BUFFER_SEND_THRESHOLD_MS}ms ({BUFFER_SEND_THRESHOLD_BYTES} bytes) threshold"
    )
    try:
        async with websockets.serve(
            handle_client, SERVER_HOST, SERVER_PORT, ping_interval=20, ping_timeout=60
        ):
            await asyncio.Future()  # Run forever
    except Exception as e:
        logger.critical(f"WebSocket server failed to start: {e}")
    finally:
        # Ensure the global engine is shut down when the server exits
        if global_coqui_engine:
            logger.info("Shutting down global CoquiEngine...")
            global_coqui_engine.shutdown()
            logger.info("Global CoquiEngine shut down.")


if __name__ == "__main__":
    asyncio.run(main())
