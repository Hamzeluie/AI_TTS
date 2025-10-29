import asyncio
from pathlib import Path
import uuid
import os
import time
import logging
from typing import List
from main_service import RedisQueueManager  # assuming your class is in a.py
from fish_engine import FishTextToSpeech  # assuming your TTS engine class is here


async def async_synthesize_voice(
    tts_engine: "FishTextToSpeech",
    text: str,
    output_dir: str,
    reference_dir: str = "./test",
) -> tuple:
    """
    Run synthesize_voice in a thread to avoid blocking the async loop.
    """
    loop = asyncio.get_running_loop()
    # Offload to thread
    result = await loop.run_in_executor(
        None,  # uses default ThreadPoolExecutor
        _synthesize_voice_sync,
        tts_engine,
        text,
        output_dir,
        reference_dir
    )
    return result

def _synthesize_voice_sync(tts_engine, text, output_dir, reference_dir):
    """Synchronous wrapper"""
    output_wav = Path(output_dir) / f"{uuid.uuid4()}.wav"
    audio, sr = tts_engine.synthesize_voice(
        text=text,
        output_codec_path_dir=output_dir,
        reference_dir=reference_dir,
        output_wav_path=str(output_wav),
        save_output=True
    )
    return str(output_wav)  # return path or audio bytes


async def tts_worker(queue_manager: RedisQueueManager, tts_engine: FishTextToSpeech):
    logger.info("TTS worker started (real synthesis).")
    while True:
        batch = await queue_manager.get_data_batch(max_batch_size=1, max_wait_time=1.0)
        if not batch:
            await asyncio.sleep(0.1)
            continue

        for req in batch:
            try:
                # Create per-request output dir
                out_dir = f"./outputs/{req.sid}"
                os.makedirs(out_dir, exist_ok=True)

                # Run TTS in thread
                wav_path = await async_synthesize_voice(
                    tts_engine=tts_engine,
                    text=req.text,
                    output_dir=out_dir,
                    reference_dir="/home/mehdi/Documents/projects/tts/fish-speech/test/ref"  # your reference voice dir
                )

                # Publish result: send WAV path or base64/audio bytes
                await queue_manager.publish_result(
                    sid=req.sid,
                    result={"audio_path": wav_path},
                    error=None
                )
                logger.info(f"✅ TTS completed for '{req.text}' → {wav_path}")

            except Exception as e:
                logger.exception(f"❌ TTS failed for {req.sid}: {e}")
                await queue_manager.publish_result(
                    sid=req.sid,
                    result=None,
                    error=str(e)
                )
                
                

# Set up logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def client_task(client_id: int, sentences: List[str], redis_url: str, queue_name: str):
    # Each client gets its OWN queue manager (for safe listening)
    req_id = sentences[0]  # Just use first sentence as req_id for simplicity
    sentences = [sentences[1]]  # Remaining are actual sentences
    queue_manager = RedisQueueManager(redis_url=redis_url, queue_name=queue_name)
    await queue_manager.initialize()

    all_results = []
    for sentence in sentences:
        words = sentence.split()
        for i, word in enumerate(words):
            # Submit via this client's manager (or you could use a shared submitter if needed)

            session_id = await queue_manager.submit_data_request(
                text=word,
                sid=req_id,
                priority= 1 if i == 0 else 2,
            )
            logger.info(f"[Client {client_id}] Submitted word '{word}' priority: {1 if i == 0 else 2}")

            # Wait for result using the SAME manager (so pubsub is isolated)
            try:
                result = await queue_manager.listen_for_result(session_id, timeout=30.0)
                if result.get('error'):
                    logger.error(f"[Client {client_id}] Error for {session_id}: {result['error']}")
                else:
                    logger.info(f"[Client {client_id}] Got result for '{word}'")
                all_results.append((word, result))
            except TimeoutError:
                logger.warning(f"[Client {client_id}] Timeout waiting for result of '{word}'")
                all_results.append((word, None))

    await queue_manager.close()
    return client_id, all_results

async def mock_worker(queue_manager: RedisQueueManager):
    """Mock worker that consumes batches and publishes dummy results."""
    logger.info("Mock worker started.")
    while True:
        batch = await queue_manager.get_data_batch(max_batch_size=8, max_wait_time=0.5)
        if not batch:
            await asyncio.sleep(0.1)
            continue
        logger.info(f"Worker processing batch of {len(batch)} requests")
        for req in batch:
            # Simulate TTS processing (e.g., fake audio = text repeated)
            fake_audio = f"[AUDIO-DATA-FOR:'{req.text}']"
            await queue_manager.publish_result(
                sid=req.sid,
                result=fake_audio,
                error=None
            )
        # Optional: break after one batch if you only want to test one round
        # break

async def main():
    redis_url = "redis://localhost:6379"
    queue_name = "inference_queue"

    # Initialize TTS engine (synchronous, do once)
    config_name = "modded_dac_vq"
    checkpoint_path = "checkpoints/openaudio-s1-mini/codec.pth"
    tts_engine = FishTextToSpeech(
        config_name=config_name,       # ← update
        checkpoint_path=checkpoint_path, # ← update
        device="cuda"
    )
    tts_engine.load_model()  # This is blocking, but OK at startup

    # Redis manager for worker
    worker_qm = RedisQueueManager(redis_url=redis_url, queue_name=queue_name)
    await worker_qm.initialize()

    # Start TTS worker
    worker_task = asyncio.create_task(tts_worker(worker_qm, tts_engine))

    # Start clients (as before)
    clients_sentences = [
        ["11*11", "Hello world"],
        ["22*22", "How are you"],
    ]
    client_tasks = [
        asyncio.create_task(client_task(i, sentences, redis_url, queue_name))
        for i, sentences in enumerate(clients_sentences)
    ]
    logger.info("Submitting first 2 sentences...")
    await asyncio.sleep(2.)  # Let them submit their first words
    # === Step 2: Submit urgent sentence (should jump ahead if priority works) ===
    logger.info("Submitting urgent sentence (first word = priority=1)...")
    urgent_future = asyncio.create_task(
        client_task(client_id=99, sentences=["33*33", "Urgent message now"], redis_url=redis_url, queue_name=queue_name)
    )
    all_futures = client_tasks + [urgent_future]
    results = await asyncio.gather(*all_futures)
    # Cleanup
    worker_task.cancel()
    await worker_qm.close()

    for client_id, word_results in results:
        logger.info(f"Client {client_id} done: {[w for w, _ in word_results]}")
        
        
async def main1():
    redis_url = "redis://localhost:6379"
    queue_name = "inference_queue"

    # Initialize worker
    worker_qm = RedisQueueManager(redis_url=redis_url, queue_name=queue_name)
    await worker_qm.initialize()
    worker_task = asyncio.create_task(mock_worker(worker_qm))

    # === Step 1: Submit first 2 sentences (early clients) ===
    early_clients = [
        ["11*11", "Hello world this is a test of a redis queue"],
        ["22*22", "How are you doing today in this fine weather"],
    ]
    early_futures = []
    for i, sentences in enumerate(early_clients):
        task = asyncio.create_task(
            client_task(client_id=i, sentences=sentences, redis_url=redis_url, queue_name=queue_name)
        )
        early_futures.append(task)

    logger.info("Submitting first 2 sentences...")
    await asyncio.sleep(2.)  # Let them submit their first words

    # === Step 2: Submit urgent sentence (should jump ahead if priority works) ===
    logger.info("Submitting urgent sentence (first word = priority=1)...")
    urgent_future = asyncio.create_task(
        client_task(client_id=99, sentences=["33*33", "Urgent message now"], redis_url=redis_url, queue_name=queue_name)
    )

    # === Step 3: Wait for all to complete ===
    all_futures = early_futures + [urgent_future]
    results = await asyncio.gather(*all_futures)

    # Cleanup
    worker_task.cancel()
    try:
        await worker_task
    except asyncio.CancelledError:
        pass
    await worker_qm.close()

    for client_id, word_results in results:
        logger.info(f"Client {client_id} completed {len(word_results)} word requests.")
        
if __name__ == "__main__":
    asyncio.run(main())