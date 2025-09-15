import time
from dia_engine import DiaEngine
from RealtimeTTS import TextToAudioStream
from multiprocessing import freeze_support


def dummy_generator():
    text = "[S1]Hello world, I am now speaking with the cloned voice characteristics. How do I sound?"
    for substr in text.split():
        yield substr + " "
    # yield text  # Yield the entire text at once for simplicity


def on_audio_chunk_callback(chunk):
    print(f"Chunk received, len: {len(chunk)}")

def main():
    engine = DiaEngine(
        model_name="nari-labs/Dia-1.6B",
        compute_dtype="float16",
        device="cuda",
        use_torch_compile=True,
    )
    # Try direct synthesis first without the stream
    # print("Testing direct synthesis...")
    # success = engine.synthesize("This is a test.")
    # print(f"Direct synthesis result: {success}")
    
    stream = TextToAudioStream(engine, muted=True).feed(dummy_generator())
    stream.play_async(
        tokenizer=None,
        language="en",
        on_audio_chunk=on_audio_chunk_callback,
        muted=True,
        output_wavfile="/home/ubuntu/ai_service/outputs/test_output.wav",
    )

    while stream.is_playing():
        time.sleep(0.1)

    # engine.shutdown()

if __name__ == '__main__':
    freeze_support()  # Required for multiprocessing on Windows
    main()
    # Initialize the engine
    
    # # Get available voices
    # voices = engine.get_voices()
    # print(f"Available voices: {voices}")

    # # Set a different voice
    # engine.set_voice("another_voice.wav")

    # # Synthesize text
    # success = engine.synthesize("Hello, this is a test of the Fish TTS engine!")

    # # The audio will be available in engine.queue
    # while not engine.queue.empty():
    #     audio_chunk = engine.queue.get()
        # Process audio_chunk (bytes)

    # Shutdown when done
    