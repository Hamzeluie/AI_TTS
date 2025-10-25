import asyncio
import json
import logging
import os
import time
from typing import Any, Dict, List

import redis.asyncio as redis
from abstract_models import *
from fish_engine import FishTextToSpeech

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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
        print("3" * 10, req)
        raw = await self.redis_client.hget(
            f"{req.agent_name}:{self.active_sessions_key}", req.sid
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
            print("2" * 10)
            await self.redis_client.hset(
                f"{req.agent_name}:{self.active_sessions_key}",
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
        print("1" * 10)
        await self.redis_client.hset(
            f"{result.agent_name}:{self.active_sessions_key}",
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
                    priority=request.priority,
                    audio=audio_sr[0],
                    sample_rate=16000,
                    created_at=request.created_at,
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

                for request in batch:
                    result_data = batch_results.get(request.sid, {})
                    await self.queue_manager.push_result(request)

                self.batch_manager.update_metrics(len(batch), processing_time)
                logger.info(
                    f"Processed batch of {len(batch)} requests in {processing_time:.3f}s"
                )
            else:
                await asyncio.sleep(0.01)
            # except Exception as e:
            #     logger.error(f"Error in batch processing loop: {e}")
            #     await asyncio.sleep(0.1)
