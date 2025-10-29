import asyncio
import logging
import os
import time
from typing import Any, Dict, List

import redis.asyncio as redis
from agent_architect.datatype_abstraction import AudioFeatures, Features, TextFeatures
from agent_architect.models_abstraction import (
    AbstractAsyncModelInference,
    AbstractInferenceServer,
    AbstractQueueManagerServer,
    DynamicBatchManager,
)
from agent_architect.session_abstraction import AgentSessions, SessionStatus
from agent_architect.utils import go_next_service
from fish_engine import FishTextToSpeech

import uvicorn
# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from fastapi import FastAPI
import uvicorn
from contextlib import asynccontextmanager


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
    ) -> List[AudioFeatures]:
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

    async def push_result(self, result: AudioFeatures):
        """Push inference result back to Redis pub/sub"""
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


class AsyncModelInference(AbstractAsyncModelInference):
    def __init__(
        self,
        model: FishTextToSpeech,
        output_codec_path_main_dir: str,
        reference_dir: str,
        max_worker: int = 4,
    ):
        super().__init__(max_worker=max_worker)
        self.model = model
        self.model.load_model()
        self.model._warm_up_models()
        self.output_codec_path_main_dir = output_codec_path_main_dir
        self.reference_dir = reference_dir

    async def _prepare_batch_inputs(
        self, batch: List[TextFeatures]
    ) -> List[AudioFeatures]:
        import os
        import shutil

        for request in batch:
            output_dir = os.path.join(self.output_codec_path_main_dir, str(request.sid))
            if os.path.exists(output_dir):
                shutil.rmtree(output_dir)
            os.makedirs(output_dir)
        return batch

    def _run_model_inference(self, batch: List[TextFeatures]) -> List:
        result = []
        for request in batch:
            output_dir = os.path.join(self.output_codec_path_main_dir, str(request.sid))
            audio_sr = self.model.synthesize_voice(
                request.text,
                output_dir,
                self.reference_dir,
                output_wav_path=f"outputs/{str(request.sid)}.wav",
                save_output=True,
            )

            result.append(
                AudioFeatures(
                    sid=request.sid,
                    agent_type=request.agent_type,
                    priority=request.priority,
                    audio=audio_sr[0].tobytes(),
                    sample_rate=16000,
                    created_at=request.created_at,
                    is_final=request.is_final,
                )
            )
        return result

    async def process_batch(self, batch: List[TextFeatures]) -> Dict[str, Any]:
        """Process a batch of requests asynchronously (final method)."""
        import time

        start_time = time.time()
        try:
            batch_inputs = await self._prepare_batch_inputs(batch)
            loop = __import__("asyncio").get_event_loop()
            results = await loop.run_in_executor(
                self.thread_pool, self._run_model_inference, batch_inputs
            )
            self._update_stats(len(batch), time.time() - start_time)
            return results
        except Exception as e:
            return await self._handle_batch_error(batch, e)


class InferenceService(AbstractInferenceServer):
    def __init__(
        self,
        model,
        output_codec_path_main_dir: str,
        reference_dir: str,
        max_worker: int = 4,
        redis_url: str = "redis://localhost:6379",
        max_batch_size: int = 16,
        max_wait_time: float = 0.1,
    ):
        super().__init__()
        self.model = model
        self.queue_manager = RedisQueueManager(redis_url)
        self.batch_manager = DynamicBatchManager(max_batch_size, max_wait_time)
        self.inference_engine = AsyncModelInference(
            model=model,
            output_codec_path_main_dir=output_codec_path_main_dir,
            reference_dir=reference_dir,
            max_worker=max_worker,
        )

    async def is_session_active(self, sid: str) -> bool:
        """Check if a session is active"""
        return await self.queue_manager.is_session_active(sid)

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
async def lifespan(app: FastAPI):
    global inference_engine, service
    logging.info("Application startup: Initializing LLM Manager...")
    config_name = "modded_dac_vq"
    checkpoint_path = "/home/ubuntu/borhan/whole_pipeline/vexu/AI_TTS/checkpoints/openaudio-s1-mini/codec.pth"
    # reference_dir = "/home/ubuntu/borhan/whole_pipeline/vexu/AI_TTS/redis_codes/test/ref"
    reference_dir = '/home/ubuntu/borhan/whole_pipeline/vexu/AI_TTS/src/test/ref'

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


app = FastAPI(lifespan=lifespan)


@app.get("/")
def read_root():
    return {"status": "LLM RAG server is running."}


# --- Main execution ---
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8103)
