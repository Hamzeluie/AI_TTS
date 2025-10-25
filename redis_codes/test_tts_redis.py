import asyncio
import os
import tempfile
from pathlib import Path
import shutil
# --- Mock TtsFeatures (normally from abstract_models) ---
from dataclasses import dataclass

# from mode1 import AsyncModelInference, InferenceService
from real_time_tts_models import *

async def main():
    # === Setup ===
    config_name = "modded_dac_vq"
    checkpoint_path = "/home/ubuntu/borhan/whole_pipeline/vexu/AI_TTS/checkpoints/openaudio-s1-mini/codec.pth"
    reference_dir = "/home/ubuntu/borhan/whole_pipeline/vexu/AI_TTS/redis_codes/test/ref"

    if not os.path.exists(reference_dir):
        raise FileNotFoundError(f"Reference dir missing: {reference_dir}")
    if not os.path.isfile(os.path.join(reference_dir, "reference_text.txt")):
        raise FileNotFoundError("Missing reference_text.txt")
    if not os.path.isfile(os.path.join(reference_dir, "reference_codec.npy")):
        raise FileNotFoundError("Missing reference_codec.npy")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    output_base = tempfile.mkdtemp(prefix="multi_user_batch_test_")
    print(f"Using temp dir: {output_base}")

    
    tts_model = FishTextToSpeech(
        config_name=config_name,
        checkpoint_path=checkpoint_path,
        device=device
    )
    service = InferenceService(
        model=tts_model,
        output_codec_path_main_dir=output_base,
        reference_dir=reference_dir,
        redis_url="redis://localhost:6379",
        max_batch_size=4,           # ← critical: match your expected batch size
        max_wait_time=0.1,          # allow 100ms window for batching
    )
    await service.start()
    print("✅ Service started – running for 10 seconds...")

    # Keep the service alive to observe queue behavior
    await asyncio.sleep(10)

    # await service.stop()
    print("✅ Service stopped")

if __name__ == "__main__":
    import torch
    asyncio.run(main())