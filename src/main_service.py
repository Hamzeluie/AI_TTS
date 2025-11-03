import asyncio
import logging
import os
import time
from typing import Any, Dict, List

import numpy as np
import redis.asyncio as redis
import uvicorn
from agent_architect.datatype_abstraction import AudioFeatures, Features, TextFeatures
from agent_architect.models_abstraction import (
    AbstractAsyncModelInference,
    AbstractInferenceServer,
    AbstractQueueManagerServer,
    DynamicBatchManager,
)
from agent_architect.session_abstraction import AgentSessions, SessionStatus
from agent_architect.utils import go_next_service
from fish_engine import FishEngine
from RealtimeTTS import TextToAudioStream
from scipy.signal import resample_poly

# from fish_text2speech import FishTextToSpeech

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI


class RedisQueueManager(AbstractQueueManagerServer):
    """
    Manages Redis-based async queue for inference requests
    """

    def __init__(
        self,
        redis_url: str = "redis://localhost:6379",
        service_name: str = "TTS",
        priorities: List[str] = ["high", "low"],
    ):
        self.redis_url = redis_url
        self.priorities = priorities
        self.service_name = service_name
        self.redis_client = None
        self.pubsub = None
        self.active_sessions_key = f"active_sessions"
        self.input_channels: List = [
            f"{self.service_name}:high",
            f"{self.service_name}:low",
        ]

    async def initialize(self):
        """Initialize Redis connection"""
        self.redis_client = await redis.from_url(self.redis_url, decode_responses=True)

    async def get_status_object(self, req: Features) -> AgentSessions:
        raw = await self.redis_client.hget(
            f"{req.agent_type}:{self.active_sessions_key}", req.sid
        )
        if raw is None:
            return None
        return AgentSessions.from_json(raw)

    async def is_session_active(self, req: Features) -> bool:
        """Check if a session is active"""
        status_obj = await self.get_status_object(req)
        if status_obj is None:
            return False
        # change status of the session to 'stop' if the session expired
        print("🤖🤖 status_obj.status", status_obj.status)
        if status_obj.is_expired():
            status_obj.status = SessionStatus.STOP
            await self.redis_client.hset(
                f"{req.agent_type}:{self.active_sessions_key}",
                req.sid,
                status_obj.to_json(),
            )
            return False
        elif status_obj.status == SessionStatus.INTERRUPT:
            print("🤖 SessionStatus.INTERRUPT")
            return False
        return True

    async def get_data_batch(
        self, max_batch_size: int = 8, max_wait_time: float = 0.1
    ) -> List[AudioFeatures]:
        batch = []
        start_time = time.time()
        while len(batch) < max_batch_size:
            elapsed = time.time() - start_time
            if elapsed >= max_wait_time and batch:
                break

            for input_channel in self.input_channels:
                # print("self.input_channels ✅", self.input_channels)
                result = await self.redis_client.brpop(input_channel, timeout=0.01)
                if result:
                    break

            if result:
                _, request_json = result
                try:
                    req = TextFeatures.from_json(request_json)
                    if not await self.is_session_active(req):
                        logger.info(f"Skipped request for stopped session: {req.sid}")
                        continue
                    batch.append(req)
                except Exception as e:
                    logger.error(f"Error parsing request: {e}")
            else:
                await asyncio.sleep(0.01)

        return batch

    async def push_result(self, result: AudioFeatures):
        """Push inference result back to Redis pub/sub"""
        if not await self.is_session_active(result):
            logger.info(f"Not pushing result for inactive session: {result.sid}")
            return

        status_obj = await self.get_status_object(result)

        status_obj.refresh_time()
        await self.redis_client.hset(
            f"{result.agent_type}:{self.active_sessions_key}",
            result.sid,
            status_obj.to_json(),
        )
        # calculate next service and queue name
        next_service = go_next_service(
            current_stage_name=self.service_name,
            service_names=status_obj.service_names,
            channels_steps=status_obj.channels_steps,
            last_channel=status_obj.last_channel,
            prioriry=result.priority,
        )
        await self.redis_client.lpush(next_service, result.to_json())
        logger.info(f"Result pushed for request {result.sid}, to {next_service}")


class AsyncModelInference:
    """
    Streaming TTS inference using FishEngine (real-time, chunked output).
    One engine → one GPU context → no thread-pool explosion.
    """

    def __init__(self, engine: FishEngine, queue_manager: AbstractQueueManagerServer):
        # NOTE: we do **not** call super() with thread_pool – we don’t need it
        self.engine = engine
        self._audio_buffers: Dict[str, bytearray] = {}  # sid → collected PCM
        self._active_tasks: Dict[str, asyncio.Task] = {}  # sid → synthesis task
        self.queue_manager: AbstractQueueManagerServer = queue_manager

    # --- Warm-up Function ---
    def warm_up_engine(self):
        """
        Performs a dummy synthesis to warm up the TTS engine.
        """
        logger.info("Warming up TTS engine...")
        print("Warming up TTS engine...")
        # Use a temporary TextToAudioStream for the warm-up
        # It needs a dummy callback, but we don't care about the output
        dummy_stream = TextToAudioStream(engine=self.engine, muted=True)
        dummy_text = "Hello. how are you today? i am your personal assistant. how can i assist you tody?"  # Short, simple text
        dummy_stream.feed(dummy_text)

        # Use a simple lambda for on_audio_chunk as we don't care about the output
        # The important part is that the synthesis process runs.
        dummy_stream.play(
            on_audio_chunk=lambda x: None,
            log_synthesized_text=False,
            fast_sentence_fragment=True,  # Make warm-up fast
            # output_wavfile="/home/ubuntu/borhan/whole_pipeline/vexu/outputs/recive/b.wav",
        )
        dummy_stream.stop()  # Clean up the dummy stream
        logger.info("TTS engine warmed up.")
        print("TTS engine warmed up.")

    async def process_batch(self, batch: List[TextFeatures]) -> List[AudioFeatures]:
        """
        Entry point from the batch manager.
        We launch a synthesis task for **each request** and return a list
        of AudioFeatures as soon as they finish.
        """
        results: List[AudioFeatures] = []

        for req in batch:
            sid = req.sid

            # 1. initialise buffer
            self._audio_buffers[sid] = bytearray()

            # 2. launch async synthesis (fire-and-forget)
            task = asyncio.create_task(self._synthesize_one(req))
            self._active_tasks[sid] = task

            # 3. wait for this request to finish
            audio_bytes = await task
            self._active_tasks.pop(sid, None)

            # 4. build final AudioFeatures
            audio_feat = AudioFeatures(
                sid=sid,
                agent_type=req.agent_type,
                priority=req.priority,
                audio=bytes(audio_bytes),
                sample_rate=16000,  # FishEngine streams 24 kHz
                created_at=req.created_at,
                is_final=req.is_final,
            )
            results.append(audio_feat)

        return results

    async def _synthesize_one(self, req: TextFeatures) -> bytearray:
        """
        Sends a synthesize command to the FishEngine worker and streams
        back the PCM chunks until ``finished`` is received.
        """
        sid = req.sid

        # Skip synthesis if text is empty or whitespace-only
        if not req.text or not req.text.strip():
            # Optionally log or handle the empty case
            logger.debug(
                f"Skipping synthesis for sid={sid}: empty or whitespace-only text"
            )
            self._audio_buffers.pop(sid, None)
            return bytearray()

        buffer = self._audio_buffers[sid]

        if self.queue_manager.is_session_active(req) is False:
            logger.info(f"Session inactive, skipping synthesis for sid={sid}")
            self._audio_buffers.pop(sid, None)
            return bytearray()

        self.engine.send_command("synthesize", {"text": req.text})
        pipe = self.engine.parent_synthesize_pipe
        while True:
            if pipe.poll(timeout=0.001):
                status, data = pipe.recv()

                if status == "success":
                    # data is already bytes (float32 PCM)
                    buffer.extend(data)

                elif status == "finished":
                    # synthesis complete
                    break

                elif status == "error":
                    raise RuntimeError(f"FishEngine error (sid={sid}): {data}")

                else:
                    logger.warning(f"Unexpected pipe status: {status}")

            else:
                # tiny sleep to keep the event loop responsive
                await asyncio.sleep(0.001)

        # Clean up
        self._audio_buffers.pop(sid, None)
        return buffer

    async def _synthesize_one1(self, req: TextFeatures) -> bytearray:
        """
        Sends a synthesize command to the FishEngine worker and streams
        back the PCM chunks until ``finished`` is received.
        """
        sid = req.sid
        buffer = self._audio_buffers[sid]
        self.engine.send_command("synthesize", {"text": req.text})
        pipe = self.engine.parent_synthesize_pipe
        while True:
            if pipe.poll(timeout=0.001):
                status, data = pipe.recv()

                if status == "success":
                    # data is already bytes (float32 PCM)
                    buffer.extend(data)

                elif status == "finished":
                    # synthesis complete
                    break

                elif status == "error":
                    raise RuntimeError(f"FishEngine error (sid={sid}): {data}")

                else:
                    logger.warning(f"Unexpected pipe status: {status}")

            else:
                # tiny sleep to keep the event loop responsive
                await asyncio.sleep(0.001)

        # Clean up
        self._audio_buffers.pop(sid, None)
        return buffer

    async def process_batch_CC(self, batch: List[TextFeatures]) -> None:
        for req in batch:
            sid = req.sid
            if sid in self._active_tasks:
                # Optionally cancel previous if interrupt
                if not await self.queue_manager.is_session_active(req):
                    self._active_tasks[sid].cancel()
            task = asyncio.create_task(self._synthesize_one(req))
            self._active_tasks[sid] = task
        # Don't await — streaming is async and decoupled

    async def _synthesize_one_CC(self, req: TextFeatures):
        sid = req.sid
        buffer = bytearray()
        # SOURCE_RATE = 24000
        SOURCE_RATE = 8000
        TARGET_RATE = 16000
        # TARGET_RATE = 8000
        SAMPLE_WIDTH = 2  # 16-bit
        THRESHOLD_MS = 150
        THRESHOLD_BYTES = int(TARGET_RATE * SAMPLE_WIDTH * (THRESHOLD_MS / 1000))
        THRESHOLD_BYTES = (THRESHOLD_BYTES // SAMPLE_WIDTH) * SAMPLE_WIDTH

        # In _synthesize_one, replace your current resampling with:
        def resample_chunk(audio_bytes):
            """Identical to a.txt's resample_audio_chunk"""
            if len(audio_bytes) == 0:
                return b""
            # FishEngine outputs int16 PCM
            audio_int16 = np.frombuffer(audio_bytes, dtype=np.int16)
            if len(audio_int16) == 0:
                return b""
            if SOURCE_RATE % TARGET_RATE == 0:
                decimated = audio_int16[:: SOURCE_RATE // TARGET_RATE]  # 24k → 8k (÷3)
                # decimated = audio_int16[::3]  # 24k → 8k (÷3)
            else:
                # Fallback interpolation
                audio_float = audio_int16.astype(np.float32) / 32768.0
                new_length = int(len(audio_float) * TARGET_RATE / SOURCE_RATE)
                if new_length == 0:
                    return b""
                old_indices = np.linspace(0, len(audio_float) - 1, len(audio_float))
                new_indices = np.linspace(0, len(audio_float) - 1, new_length)
                decimated_float = np.interp(new_indices, old_indices, audio_float)
                decimated = (decimated_float * 32767).astype(np.int16)
            return decimated.tobytes()

        # Start synthesis
        self.engine.send_command("synthesize", {"text": req.text})
        pipe = self.engine.parent_synthesize_pipe

        while True:
            if pipe.poll(timeout=0.001):
                status, data = pipe.recv()
                if status == "success":
                    resampled = resample_chunk(data)
                    if not resampled:
                        continue
                    buffer.extend(resampled)

                    # Send chunks when buffer reaches threshold
                    while len(buffer) >= THRESHOLD_BYTES:
                        chunk = bytes(buffer[:THRESHOLD_BYTES])
                        del buffer[:THRESHOLD_BYTES]
                        partial = AudioFeatures(
                            sid=sid,
                            agent_type=req.agent_type,
                            priority=req.priority,
                            audio=chunk,
                            sample_rate=TARGET_RATE,
                            created_at=req.created_at,
                            is_final=False,
                        )
                        await self.queue_manager.push_result(partial)

                elif status == "finished":
                    # Flush remaining buffer
                    if buffer:
                        final = AudioFeatures(
                            sid=sid,
                            agent_type=req.agent_type,
                            priority=req.priority,
                            audio=bytes(buffer),
                            sample_rate=TARGET_RATE,
                            created_at=req.created_at,
                            is_final=True,
                        )
                        await self.queue_manager.push_result(final)
                        buffer.clear()
                    break

                elif status == "error":
                    logger.error(f"FishEngine error (sid={sid}): {data}")
                    break

            await asyncio.sleep(0.001)


class InferenceService(AbstractInferenceServer):
    def __init__(
        self,
        model: FishEngine,
        # output_codec_path_main_dir: str,
        # reference_dir: str,
        # max_worker: int = 4,
        redis_url: str = "redis://localhost:6379",
        max_batch_size: int = 16,
        max_wait_time: float = 0.1,
    ):
        super().__init__()
        self.model = model
        self.queue_manager = RedisQueueManager(redis_url)
        self.batch_manager = DynamicBatchManager(max_batch_size, max_wait_time)
        self.inference_engine = AsyncModelInference(
            engine=model, queue_manager=self.queue_manager
        )
        self.inference_engine.warm_up_engine()

    async def is_session_active(self, req: Features) -> bool:
        """Check if a session is active"""
        return await self.queue_manager.is_session_active(req)

    async def _initialize_components(self):
        await self.queue_manager.initialize()

    async def start(self) -> None:
        """Start the inference service."""
        await self._initialize_components()
        self.is_running = True
        self.processing_task = asyncio.create_task(self._process_batches_loop())

    async def _process_batches_loop(self):
        logger.info("Starting batch processing loop")
        while self.is_running:
            # try:
            batch = await self.queue_manager.get_data_batch(
                max_batch_size=self.batch_manager.max_batch_size,
                max_wait_time=self.batch_manager.max_wait_time,
            )
            if batch:
                start_time = time.time()
                batch_results = await self.inference_engine.process_batch(batch)
                processing_time = time.time() - start_time

                for result in batch_results:
                    if await self.is_session_active(result):
                        await self.queue_manager.push_result(result)

                self.batch_manager.update_metrics(len(batch), processing_time)
                logger.info(
                    f"Processed batch of {len(batch)} requests in {processing_time:.3f}s"
                )
            else:
                await asyncio.sleep(0.01)
            # except Exception as e:
            #     logger.error(f"Error in batch processing loop: {e}")
            #     await asyncio.sleep(0.1)


service = None  # Global reference to the service for shutdown

import tempfile

import torch


@asynccontextmanager
async def lifespan_copy(app: FastAPI):
    global inference_engine, service
    logging.info("Application startup: Initializing LLM Manager...")
    config_name = "modded_dac_vq"
    checkpoint_path = "/home/ubuntu/borhan/whole_pipeline/vexu/AI_TTS/checkpoints/openaudio-s1-mini/codec.pth"
    # reference_dir = "/home/ubuntu/borhan/whole_pipeline/vexu/AI_TTS/redis_codes/test/ref"
    reference_dir = "/home/ubuntu/borhan/whole_pipeline/vexu/AI_TTS/src/test/ref"

    if not os.path.exists(reference_dir):
        raise FileNotFoundError(f"Reference dir missing: {reference_dir}")
    if not os.path.isfile(os.path.join(reference_dir, "reference_text.txt")):
        raise FileNotFoundError("Missing reference_text.txt")
    if not os.path.isfile(os.path.join(reference_dir, "reference_codec.npy")):
        raise FileNotFoundError("Missing reference_codec.npy")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)
    output_base = tempfile.mkdtemp(prefix="multi_user_batch_test_")
    print(f"Using temp dir: {output_base}")
    tts_model = FishTextToSpeech(
        config_name=config_name, checkpoint_path=checkpoint_path, device=device
    )
    service = InferenceService(
        model=tts_model,
        output_codec_path_main_dir=output_base,
        reference_dir=reference_dir,
        redis_url="redis://localhost:6379",
        max_batch_size=4,  # ← critical: match your expected batch size
        max_wait_time=0.1,  # allow 100ms window for batching
    )
    await service.start()
    logging.info("InferenceService started.")

    yield
    # Shutdown logic
    if service:
        service.is_running = False
        if service.processing_task:
            service.processing_task.cancel()
            try:
                await service.processing_task
            except asyncio.CancelledError:
                logging.info("Processing loop cancelled.")
        logging.info("InferenceService stopped.")
    logging.info("Application shutdown...")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # ------------------------------------------------------------------
    # 1. Paths (adjust to your deployment)
    # ------------------------------------------------------------------
    checkpoint_dir = "/home/ubuntu/borhan/whole_pipeline/vexu/AI_TTS/checkpoints/openaudio-s1-mini/codec.pth"
    vqgan_checkpoint = "/home/ubuntu/borhan/whole_pipeline/vexu/AI_TTS/checkpoints/openaudio-s1-mini/codec.pth"
    config_name = "modded_dac_vq"
    reference_codec = "/home/ubuntu/borhan/whole_pipeline/vexu/AI_TTS/src/test/ref/reference_codec.npy"
    reference_text = "Hello, this is the reference voice."

    device = "cuda" if torch.cuda.is_available() else "cpu"
    temp_dir = tempfile.mkdtemp(prefix="fish_tts_")

    # ------------------------------------------------------------------
    # 2. Build the engine
    # ------------------------------------------------------------------
    parent_path = "./"
    prompt = """topic but I think honestly I didn't there was no thought process to this album for me really it was all just I mean thought process in the sense of I'm going through something and trying to figure out where I'm at and what I'm feeling and what I'm going to do that but like as far as like oh this is a song that I'm going to put on a record like that wasn't it um it was really just a lot of it like before you got here I was listening to with your people and I was like, man, this is like, I was very angry. Yeah. And foundation being hurt. So yeah, so it's really just, yeah, I didn't really think about it. And I do like the idea of taking a quirky pop happy sound melodically and like the sound of the"""
    engine = FishEngine(
        checkpoint_dir=os.path.join(parent_path, "checkpoints", "openaudio-s1-mini"),
        output_dir=os.path.join(parent_path, "outputs"),
        voices_path=None,
        reference_prompt_tokens=os.path.join(parent_path, "ref_kelly.npy"),
        device="cuda",
        precision=torch.bfloat16,
        reference_prompt_text=prompt,
    )
    # engine.post_init()  # <-- spawns the worker process & waits for ready

    # ------------------------------------------------------------------
    # 3. Build the service
    # ------------------------------------------------------------------
    service = InferenceService(
        model=engine,
        redis_url="redis://localhost:6379",
        max_batch_size=4,  # small batches are fine – engine is sequential
        max_wait_time=0.05,
    )
    await service.start()
    logger.info("FishEngine TTS service ready")

    yield

    # ------------------------------------------------------------------
    # 4. Graceful shutdown
    # ------------------------------------------------------------------
    service.is_running = False
    if service.processing_task:
        service.processing_task.cancel()
    engine.shutdown()  # <-- sends shutdown command & joins process
    logger.info("FishEngine shutdown complete")


app = FastAPI(lifespan=lifespan)


@app.get("/")
def read_root():
    return {"status": "LLM RAG server is running."}


# --- Main execution ---
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8103)
