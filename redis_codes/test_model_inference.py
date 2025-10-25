import asyncio
import os
import tempfile
from pathlib import Path
import shutil
# --- Mock TtsFeatures (normally from abstract_models) ---
from dataclasses import dataclass
import time
from abstract_models import TtsFeatures
from fish_engine import FishTextToSpeech
from real_time_tts_models import AsyncModelInference, InferenceService

async def main1():
    # === CONFIGURATION ===
    config_name = "modded_dac_vq"
    checkpoint_path = "checkpoints/openaudio-s1-mini/codec.pth"
    reference_dir = "./test/ref"              # Must exist with text + codec
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Create output dir
    output_dir = tempfile.mkdtemp(prefix="/home/mehdi/Documents/projects/tts/fish-speech/tts_output_")
    print(f"Using output dir: {output_dir}")

    # === INIT MODEL ===
    tts_model = FishTextToSpeech(
        config_name=config_name,
        checkpoint_path=checkpoint_path,
        device=device
    )
    
    # === WRAP IN AsyncModelInference ===
    async_infer = AsyncModelInference(
        model=tts_model,
        output_codec_path_main_dir=output_dir,
        reference_dir=reference_dir,
        max_worker=2
    )

    # === PREPARE TEST REQUESTS ===
    test_sentences = [
        "یہ پہلا جملہ ہے۔",
        "یہ دوسرا جملہ ہے۔",
        "تیسرا جملہ آواز کی جانچ کے لیے۔"
    ]

    batch = [
        TtsFeatures(sid=f"req_{i}", text=sent)
        for i, sent in enumerate(test_sentences)
    ]

    # === PROCESS BATCH ===
    print("Processing batch...")
    results = await async_infer.process_batch(batch)

    # === PRINT RESULTS ===
    for sid, res in results.items():
        if 'error' in res:
            print(f"[ERROR] {sid}: {res['error']}")
        else:
            audio, sr = res['result']
            duration = len(audio) / sr
            print(f"[OK] {sid}: generated {duration:.2f}s audio @ {sr} Hz")

    # === SHOW STATS ===
    print("\nInference Stats:")
    for k, v in async_infer.stats.items():
        print(f"  {k}: {v:.3f}" if isinstance(v, float) else f"  {k}: {v}")

    # Cleanup (optional)
    # shutil.rmtree(output_dir)

async def main2():
    # === CONFIGURATION ===
    config_name = "modded_dac_vq"  # Must match your Hydra config filename (without .yaml)
    checkpoint_path = "checkpoints/openaudio-s1-mini/codec.pth"
    reference_dir = "./test/ref"  # Must contain: reference_text.txt + reference_codec.npy

    if not os.path.exists(reference_dir):
        raise FileNotFoundError(f"Reference dir missing: {reference_dir}")
    if not os.path.isfile(os.path.join(reference_dir, "reference_text.txt")):
        raise FileNotFoundError(f"Missing {reference_dir}/reference_text.txt")
    if not os.path.isfile(os.path.join(reference_dir, "reference_codec.npy")):
        raise FileNotFoundError(f"Missing {reference_dir}/reference_codec.npy")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    output_dir = tempfile.mkdtemp(prefix="tts_output_")
    print(f"Using output dir: {output_dir}")

    # === INIT MODEL ===
    tts_model = FishTextToSpeech(
        config_name=config_name,
        checkpoint_path=checkpoint_path,
        device=device
    )

    # === INIT INFERENCE WRAPPER ===
    async_infer = AsyncModelInference(
        model=tts_model,
        output_codec_path_main_dir=output_dir,
        reference_dir=reference_dir,
        max_worker=2
    )

    # === TEST SENTENCES (Urdu) ===
    test_sentences = [
        "یہ پہلا جملہ ہے۔",
        "یہ دوسرا جملہ ہے۔",
        "تیسرا جملہ آواز کی جانچ کے لیے۔"
    ]

    # === CREATE BATCH WITH created_at SET ===
    batch = [
        TtsFeatures(
            sid=f"req_{i}",
            text=sent,
            priority=1,
            created_at=time.time()
        )
        for i, sent in enumerate(test_sentences)
    ]

    # === PROCESS BATCH ===
    print("Processing batch...")
    results = await async_infer.process_batch(batch)

    # === REPORT RESULTS ===
    for sid, res in results.items():
        if 'error' in res:
            print(f"[ERROR] {sid}: {res['error']}")
        else:
            audio_list, sr = res['result']
            duration = len(audio_list) / sr
            print(f"[OK] {sid}: generated {duration:.2f}s audio @ {sr} Hz")

    # === PRINT STATS ===
    stats = async_infer.stats
    print("\nInference Stats:")
    print(f"  Total batches: {stats['total_batches']}")
    print(f"  Total requests: {stats['total_requests']}")
    print(f"  Avg batch size: {stats['avg_batch_size']:.2f}")
    print(f"  Avg inference time: {stats['avg_inference_time']:.3f}s")

async def main3():
    # === Setup ===
    config_name = "modded_dac_vq"
    checkpoint_path = "checkpoints/openaudio-s1-mini/codec.pth"
    reference_dir = "./test/ref"

    if not os.path.exists(reference_dir):
        raise FileNotFoundError(f"Reference dir missing: {reference_dir}")
    if not os.path.isfile(os.path.join(reference_dir, "reference_text.txt")):
        raise FileNotFoundError("Missing reference_text.txt")
    if not os.path.isfile(os.path.join(reference_dir, "reference_codec.npy")):
        raise FileNotFoundError("Missing reference_codec.npy")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    output_base = tempfile.mkdtemp(prefix="dynamic_batch_test_")
    print(f"Using temp dir: {output_base}")

    # === Init model ===
    tts_model = FishTextToSpeech(
        config_name=config_name,
        checkpoint_path=checkpoint_path,
        device=device
    )

    # === Start service with short wait time to enable batching ===
    service = InferenceService(
        model=tts_model,
        output_codec_path_main_dir=output_base,
        reference_dir=reference_dir,
        redis_url="redis://localhost:6379",
        max_batch_size=8,
        max_wait_time=0.05,  # 50ms — requests arriving within this window get batched
        queue_name="test_dynamic_queue"
    )

    await service.start()
    print("✅ Inference service started")

    try:
        # === Simulate staggered requests ===
        sentences = [
            "یہ پہلا جملہ ہے۔",
            "یہ دوسرا جملہ ہے۔",
            "تیسرا جملہ آواز کی جانچ کے لیے۔",
            "چوتھا جملہ بیچ میں شامل ہوا۔",
            "پانچواں جملہ آخر میں آیا۔"
        ]

        async def submit_with_delay(idx, text, delay_ms, priority):
            await asyncio.sleep(delay_ms / 1000.0)
            sid = f"req_{idx}_{int(time.time() * 1000)}"
            print(f"[{time.time():.3f}] Submitting {sid}")
            try:
                result = await service.predict(input_data=text, sid=sid, priority=priority, timeout=60.0)
                audio, sr = result
                duration = len(audio) / sr
                print(f"[{time.time():.3f}] ✅ {sid} done ({duration:.2f}s)")
            except Exception as e:
                print(f"[{time.time():.3f}] ❌ {sid} failed: {e}")

        # Launch staggered tasks (e.g., 0ms, 15ms, 30ms, 40ms, 45ms → all within 50ms window)
        delays_ms = [0, 15, 3000, 40, 45]
        tasks = [
            asyncio.create_task(submit_with_delay(i, sent, delay, 1 if i==2 else 2))
            for i, (sent, delay) in enumerate(zip(sentences, delays_ms))
        ]

        print(f"\n🚀 Submitting {len(tasks)} requests with staggered delays...")
        await asyncio.gather(*tasks)

        # === Show stats ===
        stats = service.inference_engine.stats
        print("\n📊 Final Stats:")
        print(f"  Total batches: {stats['total_batches']}")
        print(f"  Total requests: {stats['total_requests']}")
        print(f"  Avg batch size: {stats['avg_batch_size']:.2f}")
        print(f"  Avg inference time: {stats['avg_inference_time']:.3f}s")

        # ✅ Success condition: avg_batch_size > 1.0 means dynamic batching worked
        if stats['avg_batch_size'] > 1.0:
            print("\n🎉 Dynamic batching confirmed! Requests were merged into batches.")
        else:
            print("\n⚠️  No batching observed. Check max_wait_time or request timing.")

    finally:
        await service.stop()
        print("🛑 Service stopped")

async def main4():
    # === Setup ===
    config_name = "modded_dac_vq"
    checkpoint_path = "checkpoints/openaudio-s1-mini/codec.pth"
    reference_dir = "./test/ref"

    if not os.path.exists(reference_dir):
        raise FileNotFoundError(f"Reference dir missing: {reference_dir}")
    if not os.path.isfile(os.path.join(reference_dir, "reference_text.txt")):
        raise FileNotFoundError("Missing reference_text.txt")
    if not os.path.isfile(os.path.join(reference_dir, "reference_codec.npy")):
        raise FileNotFoundError("Missing reference_codec.npy")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    output_base = tempfile.mkdtemp(prefix="dynamic_batch_test_")
    print(f"Using temp dir: {output_base}")

    # === Init model ===
    tts_model = FishTextToSpeech(
        config_name=config_name,
        checkpoint_path=checkpoint_path,
        device=device
    )

    # === Start service with short wait time to enable batching ===
    service = InferenceService(
        model=tts_model,
        output_codec_path_main_dir=output_base,
        reference_dir=reference_dir,
        redis_url="redis://localhost:6379",
        max_batch_size=8,
        max_wait_time=0.05,  # 50ms — requests arriving within this window get batched
        queue_name="test_dynamic_queue"
    )
    try:
        await service.start()
        print("✅ Inference service started")
        await service.start_session("client_1")
        await service.start_session("client_2")
        await service.start_session("client_3")
        await service.start_session("client_4")
        
        
        result1 = await service.predict(input_data="Hello world client 1",sid="client_1", priority=1)
        result2 = await service.predict(input_data="how are you client 2",sid="client_2", priority=1)
        result3 = await service.predict(input_data="hello assistant client 3",sid="client_3", priority=1)
        result1 = await service.predict(input_data="whats up? client 1",sid="client_1", priority=2)
        result2 = await service.predict(input_data="my name is mehdi client 2",sid="client_2", priority=2)
        result4 = await service.predict(input_data="now i am comming client 4",sid="client_4", priority=1)
        await service.stop_session("client_3")
        try:
            result3 = await service.predict(input_data="i like you in there client 3",sid="client_3", priority=2)
        except Exception as e:
            print(f"✅ Expected error for stopped session client_3: {e}")
        result4 = await service.predict(input_data="this is new client 4",sid="client_4", priority=2)


        # === Show stats ===
        stats = service.inference_engine.stats
        print("\n📊 Final Stats:")
        print(f"  Total batches: {stats['total_batches']}")
        print(f"  Total requests: {stats['total_requests']}")
        print(f"  Avg batch size: {stats['avg_batch_size']:.2f}")
        print(f"  Avg inference time: {stats['avg_inference_time']:.3f}s")

        # ✅ Success condition: avg_batch_size > 1.0 means dynamic batching worked
        if stats['avg_batch_size'] > 1.0:
            print("\n🎉 Dynamic batching confirmed! Requests were merged into batches.")
        else:
            print("\n⚠️  No batching observed. Check max_wait_time or request timing.")

    finally:
        await service.stop()
        print("🛑 Service stopped")

async def main5():
    # === Setup ===
    config_name = "modded_dac_vq"
    checkpoint_path = "checkpoints/openaudio-s1-mini/codec.pth"
    reference_dir = "./test/ref"

    if not os.path.exists(reference_dir):
        raise FileNotFoundError(f"Reference dir missing: {reference_dir}")
    if not os.path.isfile(os.path.join(reference_dir, "reference_text.txt")):
        raise FileNotFoundError("Missing reference_text.txt")
    if not os.path.isfile(os.path.join(reference_dir, "reference_codec.npy")):
        raise FileNotFoundError("Missing reference_codec.npy")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    output_base = tempfile.mkdtemp(prefix="dynamic_batch_test_")
    print(f"Using temp dir: {output_base}")

    # === Init model ===
    tts_model = FishTextToSpeech(
        config_name=config_name,
        checkpoint_path=checkpoint_path,
        device=device
    )

    # === Start service with short wait time to enable batching ===
    service = InferenceService(
        model=tts_model,
        output_codec_path_main_dir=output_base,
        reference_dir=reference_dir,
        redis_url="redis://localhost:6379",
        max_batch_size=8,
        max_wait_time=0.05,  # 50ms — requests arriving within this window get batched
        queue_name="test_dynamic_queue"
    )
    try:
        await service.start()
        print("✅ Inference service started")

        # Start sessions
        await asyncio.gather(
            service.start_session("client_1"),
            service.start_session("client_2"),
            service.start_session("client_3"))

        # === Submit requests in parallel to enable batching ===
        # Phase 1: High-priority requests (priority=1)
        tasks_phase1 = [
            service.predict("Hello world client 1", "client_1", priority=1),
            service.predict("how are you client 2", "client_2", priority=1),
            service.predict("hello assistant client 3", "client_3", priority=1),
        ]

        # Launch all phase 1 tasks concurrently
        results_phase1 = await asyncio.gather(*tasks_phase1, return_exceptions=True)

        await asyncio.gather(service.start_session("client_4"))
        # Phase 2: Stop client_3 and submit low-priority requests (priority=2)
        await service.stop_session("client_3")

        tasks_phase2 = [
            service.predict("whats up? client 1", "client_1", priority=2),
            service.predict("my name is mehdi client 2", "client_2", priority=2),
            # Intentionally try to use stopped session — will fail
            service.predict("i like you in there client 3", "client_3", priority=2),
            service.predict("this is new client 4", "client_4", priority=1),
        ]

        results_phase2 = await asyncio.gather(*tasks_phase2, return_exceptions=True)

        # Handle the expected error for client_3
        for i, res in enumerate(results_phase2):
            if isinstance(res, Exception) and "client_3" in str(res):
                print(f"✅ Expected error for stopped session client_3: {res}")
            elif isinstance(res, Exception):
                print(f"⚠️ Unexpected error: {res}")

        # === Show stats ===
        stats = service.inference_engine.stats
        print("\n📊 Final Stats:")
        print(f"  Total batches: {stats['total_batches']}")
        print(f"  Total requests: {stats['total_requests']}")
        print(f"  Avg batch size: {stats['avg_batch_size']:.2f}")
        print(f"  Avg inference time: {stats['avg_inference_time']:.3f}s")

        # ✅ Success condition: avg_batch_size > 1.0 means dynamic batching worked
        if stats['avg_batch_size'] > 1.0:
            print("\n🎉 Dynamic batching confirmed! Requests were merged into batches.")
        else:
            print("\n⚠️  No batching observed. Check max_wait_time or request timing.")

    finally:
        await service.stop()
        print("🛑 Service stopped")

async def main():
    # === Setup ===
    config_name = "modded_dac_vq"
    checkpoint_path = "checkpoints/openaudio-s1-mini/codec.pth"
    reference_dir = "./test/ref"

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
        queue_name="test_multi_user_batch"
    )

    try:
        await service.start()
        print("✅ Service started")

        # Start sessions
        clients = ["client_1", "client_2", "client_3", "client_4"]
        await asyncio.gather(*[service.start_session(sid) for sid in clients])

        # Define requests per your spec:
        # client_1: 2 sentences
        # client_2: 2 sentences
        # client_3: 1 sentence
        # client_4: 4 sentences
        requests = [
            ("Hello from client 1 - A", "client_1", 2),
            ("Hello from client 1 - B", "client_1", 1),
            ("Hi from client 2 - A", "client_2", 1),
            ("Hi from client 2 - B", "client_2", 2),
            ("Greetings from client 3", "client_3", 1),
            ("Msg 1 from client 4", "client_4", 2),
            ("Msg 2 from client 4", "client_4", 2),
            ("Msg 3 from client 4", "client_4", 2),
            ("Msg 4 from client 4", "client_4", 2),
        ]

        # Submit ALL requests CONCURRENTLY to maximize batching chance
        tasks = [
            service.predict(text, sid, priority=prio) for text, sid, prio in requests
        ]

        print(f"🚀 Submitting {len(requests)} requests from 4 clients...")
        start_time = asyncio.get_event_loop().time()
        results = await asyncio.gather(*tasks, return_exceptions=True)
        elapsed = asyncio.get_event_loop().time() - start_time

        print(f"✅ All {len([r for r in results if not isinstance(r, Exception)])} requests completed in {elapsed:.2f}s")

        # Print stats
        await asyncio.sleep(20.2)  # Let final batch stats settle
        stats = service.inference_engine.stats
        print("\n📊 Final Batching Stats:")
        print(f"  Total batches: {stats['total_batches']}")
        print(f"  Total requests: {stats['total_requests']}")
        print(f"  Avg batch size: {stats['avg_batch_size']:.2f}")

        # Validate expectations
        expected_total = 9
        if stats['total_requests'] != expected_total:
            print(f"⚠️  Expected {expected_total} requests, got {stats['total_requests']}")

        # Ideal batching with max_batch_size=4 and concurrent submission:
        # Batch 1: 4 reqs (c1×2 + c2×2)
        # Batch 2: 4 reqs (c3×1 + c4×3)
        # Batch 3: 1 req  (c4×1)
        # → total_batches = 3, avg_batch_size = 9/3 = 3.0

        if stats['total_batches'] == 3 and abs(stats['avg_batch_size'] - 3.0) < 0.1:
            print("🎉 SUCCESS: Batching matched expected pattern!")
        else:
            print("🔍 Batching occurred, but pattern differs (timing or model warm-up may affect it)")

        # Optional: print per-request results
        for i, (text, sid, _) in enumerate(requests):
            if isinstance(results[i], Exception):
                print(f"❌ {sid}: {results[i]}")
            else:
                print(f"✅ {sid}: processed")

    finally:
        await service.stop()
        shutil.rmtree(output_base, ignore_errors=True)
        print("🛑 Service stopped and temp dir cleaned")


if __name__ == "__main__":
    import torch
    asyncio.run(main())