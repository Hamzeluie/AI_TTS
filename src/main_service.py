import asyncio
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import redis.asyncio as redis
import torchaudio
import uvicorn
from agent_architect.datatype_abstraction import (
    Features,
    TextAudioFeatures,
    TextFeatures,
)
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
import wave
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI

PUNCTUATION_MARKS = {".", "!", "?", ";", ":", "\n"}
MAX_BUFFER_WORDS = 1  # or use char limit like MAX_BUFFER_CHARS = 100
MAX_BUFFER_CHARS = 500000000
# Audio configuration
COQUI_SAMPLE_RATE = 24000  # CoquiEngine outputs at 24kHz
TARGET_SAMPLE_RATE = 8000
TARGET_SAMPLE_WIDTH_BYTES = 2  # 16-bit PCM

# Manual buffer configuration for increased fluency
BUFFER_SEND_THRESHOLD_MS = (
    150  # Send audio when buffer accumulates this many milliseconds (e.g., 150-200ms)
)
# Calculate buffer size in bytes, ensuring it's a multiple of the sample width
BUFFER_SEND_THRESHOLD_BYTES = int(
    TARGET_SAMPLE_RATE * TARGET_SAMPLE_WIDTH_BYTES * (BUFFER_SEND_THRESHOLD_MS / 1000.0)
)
BUFFER_SEND_THRESHOLD_BYTES = (
    BUFFER_SEND_THRESHOLD_BYTES // TARGET_SAMPLE_WIDTH_BYTES
) * TARGET_SAMPLE_WIDTH_BYTES

STOP_WORD = "bye"


def save_wav(pcm_bytes: bytes, sample_rate: int, output_path: Path):
    audio_np = np.frombuffer(pcm_bytes, dtype=np.float32)
    audio_np = np.clip(audio_np, -1.0, 1.0)
    audio_int16 = (audio_np * 32767).astype(np.int16)
    with wave.open(str(output_path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio_int16.tobytes())
    logger.info(f"Saved {output_path}")


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
        print("🤖 SessionStatus.Status", status_obj.status)
        if status_obj.is_expired():
            status_obj.status = SessionStatus.STOP
            await self.redis_client.hset(
                f"{req.agent_type}:{self.active_sessions_key}",
                req.sid,
                status_obj.to_json(),
            )
            return False
        elif status_obj.status == SessionStatus.INTERRUPT:
            return False
        return True

    async def get_data_batch(
        self, max_batch_size: int = 8, max_wait_time: float = 0.1
    ) -> List[TextAudioFeatures]:
        batch = []
        start_time = time.time()
        while len(batch) < max_batch_size:
            elapsed = time.time() - start_time
            if elapsed >= max_wait_time and batch:
                break

            for input_channel in self.input_channels:
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

    async def push_result(self, result: TextAudioFeatures):
        """Push inference result back to Redis pub/sub"""
        if not await self.is_session_active(result):
            logger.info(f"Not pushing result for inactive session: {result.sid}")
            return

        """
        # Save audio file before processing
        print("💾 Saving audio file for session:", result.sid)
        # Save audio to directory (minimal version)
        audio_dir = Path("/home/ubuntu/borhan/whole_pipeline/vexu/AI_TTS/outputs/temp")
        audio_dir.mkdir(exist_ok=True)
        file_count = len(list(audio_dir.glob("audio_*.wav"))) + 1
        filename = f"audio_{file_count:06d}.wav"
        save_wav(result.audio, 24000, str(audio_dir / filename))
        logger.info(f"Audio saved as {filename}")
        print("💾 Saved audio file for session:", result.sid)
        """

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
            sid=result.sid,
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
        self._audio_counters: Dict[str, int] = {}  # sid → word count
        self.queue_manager: AbstractQueueManagerServer = queue_manager
        self._buffer_lock = asyncio.Lock()

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
        dummy_text = "Hello. how are you today?"  # Short, simple text
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

    async def process_single(self, req: TextFeatures) -> List[TextAudioFeatures]:
        """
        Entry point from the batch manager.
        We launch a synthesis task for **each request** and return a list
        of TextAudioFeatures as soon as they finish.
        """
        sid = req.sid

        # 1. initialise buffer
        self._audio_buffers[sid] = bytearray()
        self._audio_counters[sid] = 0

        # 2. launch async synthesis (fire-and-forget)
        if await self.queue_manager.is_session_active(req) is False:
            logger.info(f"❌❌ Session inactive, skipping synthesis for sid={sid}")
            self._audio_buffers.pop(sid, None)
            self._audio_counters.pop(sid, None)
            await asyncio.sleep(1.0)
            return None

        task = asyncio.create_task(self._synthesize_one(req))
        # 3. wait for this request to finish
        self._audio_buffers[sid] = await task

        if await self.queue_manager.is_session_active(req) is False:
            logger.info(f"❌ Session inactive, skipping synthesis for sid={sid}")
            self._audio_buffers.pop(sid, None)
            self._audio_counters.pop(sid, None)
            await asyncio.sleep(1.0)
            return None

        if self._audio_counters[sid] >= MAX_BUFFER_WORDS:
            # 4. build final TextAudioFeatures
            audio_feat = TextAudioFeatures(
                sid=sid,
                agent_type=req.agent_type,
                priority=req.priority,
                audio=bytes(self._audio_buffers[sid]),
                text=req.text,
                sample_rate=16000,  # FishEngine streams 24 kHz
                created_at=req.created_at,
                is_final=req.is_final,
            )
            del self._audio_buffers[sid]
            del self._audio_counters[sid]
            return audio_feat

        if req.is_final:
            audio_feat = TextAudioFeatures(
                sid=sid,
                agent_type=req.agent_type,
                priority=req.priority,
                audio=bytes(self._audio_buffers[sid]),
                text=req.text,
                sample_rate=16000,  # FishEngine streams 24 kHz
                created_at=req.created_at,
                is_final=req.is_final,
            )

            del self._audio_buffers[sid]
            del self._audio_counters[sid]
            return audio_feat

        return None

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
        # pass voice clone path and text to engine.
        self.engine.send_command("synthesize", {"text": req.text})
        pipe = self.engine.parent_synthesize_pipe
        while True:
            if pipe.poll(timeout=0.001):
                status, data = pipe.recv()

                if status == "success":
                    async with self._buffer_lock:
                        # data is already bytes (float32 PCM)
                        buffer.extend(data)
                        if req.text not in PUNCTUATION_MARKS:
                            self._audio_counters[sid] += 1

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

    async def process_batch(self, batch: List[TextFeatures]) -> List[TextAudioFeatures]:
        """
        Entry point from the batch manager.
        We launch a synthesis task for **each request** and return a list
        of TextAudioFeatures as soon as they finish.
        """

        results: List[TextAudioFeatures] = []
        for req in batch:
            sid = req.sid
            # 1. initialise buffer
            self._audio_buffers[sid] = bytearray()

            # 2. launch async synthesis (fire-and-forget)
            if self.queue_manager.is_session_active(req) is False:
                logger.info(f"❌ Session inactive, skipping synthesis for sid={sid}")
                self._audio_buffers.pop(sid, None)
                break
            task = asyncio.create_task(self._synthesize_one(req))
            # 3. wait for this request to finish
            # audio_bytes = await task
            self._audio_buffers[sid] = await task

            # condition buffer self._audio_buffers[sid]
            # if len(self._audio_buffers[sid]) >= BUFFER_SEND_THRESHOLD_BYTES:

            if self.queue_manager.is_session_active(req) is False:
                logger.info(f"❌ Session inactive, skipping synthesis for sid={sid}")
                self._audio_buffers.pop(sid, None)
                break

            # Extract a chunk of the threshold size
            # 4. build final AudioFeatures
            print("☠️", bytes(self._audio_buffers[sid][:BUFFER_SEND_THRESHOLD_BYTES]))
            audio_feat = TextAudioFeatures(
                sid=sid,
                agent_type=req.agent_type,
                priority=req.priority,
                audio=bytes(self._audio_buffers[sid][:BUFFER_SEND_THRESHOLD_BYTES]),
                text=req.text,
                sample_rate=16000,  # FishEngine streams 24 kHz
                created_at=req.created_at,
                is_final=req.is_final,
            )
            results.append(audio_feat)
            del self._audio_buffers[sid][:BUFFER_SEND_THRESHOLD_BYTES]

        return results


class InferenceService(AbstractInferenceServer):
    def __init__(
        self,
        model: FishEngine,
        # output_codec_path_main_dir: str,
        # reference_dir: str,
        # max_worker: int = 4,
        redis_url: str = "redis://localhost:6379",
        max_batch_size: int = 1,
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

    async def end_call(self, result: TextFeatures):
        """Push inference result back to Redis pub/sub"""
        if STOP_WORD in result.text.lower():
            print("🛑 Stop word detected. Ending session:", result.sid)
            status_obj = await self.queue_manager.get_status_object(result)
            status_obj.status = SessionStatus.STOP
            await self.queue_manager.redis_client.hset(
                f"{result.agent_type}:{self.queue_manager.active_sessions_key}",
                result.sid,
                status_obj.to_json(),
            )

    async def _process_batches_loop(self):
        logger.info("Starting batch processing loop")
        while self.is_running:
            # try:
            batch = await self.queue_manager.get_data_batch(
                max_batch_size=self.batch_manager.max_batch_size,
                max_wait_time=self.batch_manager.max_wait_time,
            )
            if batch:

                for req in batch:
                    start_time = time.time()
                    result = await self.inference_engine.process_single(req)
                    processing_time = time.time() - start_time

                    if result is not None:
                        await self.queue_manager.push_result(result)
                        await self.end_call(req)

                        self.batch_manager.update_metrics(len(batch), processing_time)
                        logger.info(
                            f"Processed batch of {len(batch)} requests in {processing_time:.3f}s"
                        )

                # start_time = time.time()
                # batch_results = await self.inference_engine.process_batch(batch)
                # processing_time = time.time() - start_time

                # for result in batch_results:
                #     await self.queue_manager.push_result(result)
                #     # await self.end_call(req)

                # self.batch_manager.update_metrics(len(batch), processing_time)
                # logger.info(
                #     f"Processed batch of {len(batch)} requests in {processing_time:.3f}s"
                # )

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
    parent_path = "/home/ubuntu/borhan/whole_pipeline/vexu/AI_TTS"
    prompt = """topic but I think honestly I didn't there was no thought process to this album for me really it was all just I mean thought process in the sense of I'm going through something and trying to figure out where I'm at and what I'm feeling and what I'm going to do that but like as far as like oh this is a song that I'm going to put on a record like that wasn't it um it was really just a lot of it like before you got here I was listening to with your people and I was like, man, this is like, I was very angry. Yeah. And foundation being hurt. So yeah, so it's really just, yeah, I didn't really think about it. And I do like the idea of taking a quirky pop happy sound melodically and like the sound of the"""
    # prompt = """ماما من الناس اللي تدعمني بكل خطوة في حياتي  عشان كذا ما عندي جانب النقص اللي بحس فيه للأسف  اللي بحسوا فيه البعض وبسبب أهاليهم  أنه أنتي ناقصة لأنك لسه ما تزوجتي  أكيد أمي زيها زي أي أم ثانية  تتمنى تشوفني مستقرة في يوم والأيام في علاقة  ولكن لما يكون الوقت مناسب  الاختيار صحيح  مش لي مجرد أنه أنا ألبس فستان أبيض وتشوفني عروسة  ولما نحب نكتشف أنه احنا حبينا شخص في حياتنا  ما كنا نتخيل نحب سلبية في شخصيته مثلاً"""
    # prompt = """أنا مامتي كانت بتلعب سباحة كانت بطلة كبيرة جدا فخلوني أخش ألعب سباحة فلما ألعب سباحة من وانا عندي أربع سنين لغاية تمانْتاشر سنة كل يوم بصحة أربعة الصبح عشان خمسة الصبح أكون في المية فبقيت عندي شخصية تركيبة معينة واخد بالك فطفولتي ما كانتش أن أنا في الجنينة وبلعب بالحاجات وفي رفاهية الوقت أو رفاهية إنك ما عملتش ده النهاردة خلاص بكرة إن شاء الله لا كان فيه شدة وكان فيه التزام علي كان لازم بالذات في الرياضة كان لازم أكون شطرة"""

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
        max_batch_size=1,  # small batches are fine – engine is sequential
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
