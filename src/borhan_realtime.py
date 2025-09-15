import asyncio
import functools
import logging
import os
import queue
import threading
from pathlib import Path

import numpy as np
import torch
import websockets
import yaml

# Configure for headless server environment
os.environ["ALSA_PCM_DEVICE"] = "0"
os.environ["ALSA_PCM_CARD"] = "0"
os.environ["SDL_AUDIODRIVER"] = "dummy"

# Configure logging
parent_path = Path(__file__).parents[1]
os.makedirs(os.path.join(parent_path, "outputs"), exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    filename=os.path.join(parent_path, "outputs", "tts_server.log"),
)
logger = logging.getLogger(__name__)

# Load YAML configuration
yaml_path = Path(__file__).parents[2] / "config.yaml"
print(yaml_path)
yaml_config = yaml.safe_load(open(yaml_path, "r"))

# Import after setting up environment
from RealtimeTTS import TextToAudioStream

# This needs to be in the main guard to avoid multiprocessing issues
if __name__ == "__main__":
    # Import FishEngine inside the main guard
    from fish_engine import FishEngine

    # Initialize the engine
    prompt = """topic but I think honestly I didn't there was no thought process to this album for me really it was all just I mean thought process in the sense of I'm going through something and trying to figure out where I'm at and what I'm feeling and what I'm going to do that but like as far as like oh this is a song that I'm going to put on a record like that wasn't it um it was really just a lot of it like before you got here I was listening to with your people and I was like, man, this is like, I was very angry. Yeah. And foundation being hurt. So yeah, so it's really just, yeah, I didn't really think about it. And I do like the idea of taking a quirky pop happy sound melodically and like the sound of the"""

    print("Initializing FishEngine...")
    global_coqui_engine = FishEngine(
        checkpoint_dir=os.path.join(parent_path, "checkpoints", "openaudio-s1-mini"),
        output_dir=os.path.join(parent_path, "outputs"),
        voices_path=None,
        reference_prompt_tokens=os.path.join(parent_path, "ref_kelly.npy"),
        device="cuda",
        precision=torch.bfloat16,
        reference_prompt_text=prompt,
    )

    print("Creating TextToAudioStream...")
    stream = TextToAudioStream(
        engine=global_coqui_engine,
        muted=True,
        log_characters=False,
    )

    # Test text
    test_text = "Hello, this is a test to see if we can restart the stream after stopping."

    print("Starting stream for the first time...")
    stream.feed(test_text)
    stream.play_async()  # Start playing asynchronously

    # Wait a bit
    import time
    time.sleep(2)

    print("Stopping stream...")
    stream.stop()  # Stop the stream

    print("Waiting a moment...")
    time.sleep(1)

    print("Attempting to restart stream...")
    try:
        # Try to feed new text and restart
        stream.feed("This is text after stopping. If you hear this, restart worked!")
        stream.play_async()
        
        # Wait a bit to hear the result
        time.sleep(0.5)
        stream.stop()
        print("Restart appears to have worked!")
        stream.feed("Another test")
        stream.play_async()
        
    except Exception as e:
        print(f"Error when trying to restart: {e}")
        print("The stream cannot be restarted after stopping.")
        
        # Try creating a new stream instance instead
        print("Trying with a new stream instance...")
        try:
            new_stream = TextToAudioStream(
                engine=global_coqui_engine,
                muted=True,
                log_characters=False,
            )
            
            new_stream.feed("This is from a new stream instance after the first one failed to restart.")
            new_stream.play_async()
            time.sleep(3)
            new_stream.stop()
            print("New stream instance worked!")
            
        except Exception as e2:
            print(f"Error with new stream instance too: {e2}")
    
    print("Test completed.")