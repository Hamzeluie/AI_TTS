#!/usr/bin/env python3
import asyncio
import logging
import os
import sys
import time
from typing import Any, Dict

import redis.asyncio as redis

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent_architect.datatype_abstraction import Features, TextFeatures
from agent_architect.session_abstraction import AgentSessions, SessionStatus

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Redis configuration (must match your RAG service)
INPUT_CHANNELS = ["RAG:high", "RAG:low"]
OUTPUT_CHANNELS = ["TTS:high", "TTS:low"]
ACTIVE_SESSIONS_KEY = "call:active_sessions"

SAMPLE_RATE = int(os.getenv("VAD_SAMPLE_RATE", 16000))
AGENT_NAME = "call"
SERVICE_NAMES = ["VAD", "STT", "RAG", "TTS"]
CHANNEL_STEPS = {
    "VAD": ["input"],
    "STT": ["high", "low"],
    "RAG": ["high", "low"],
    "TTS": ["high", "low"],
}
INPUT_CHANNEL = f"{SERVICE_NAMES[0]}:{CHANNEL_STEPS[SERVICE_NAMES[0]][0]}"
OUTPUT_CHANNEL = f"{AGENT_NAME.lower()}:output"


async def publish_stt_requests(redis_client, num_requests: int = 5):
    sids = [f"test_sid_{i}" for i in range(num_requests)]
    txt_objects = []
    sentences = [
        "Hello, how can I help you?",
        "I love music.",]
    # Mark sessions as active
    for sid, sentence in zip(sids, sentences):
        status = AgentSessions(
            sid=sid,
            agent_type=AGENT_NAME,
            agent_id="AGENT_ID",
            service_names=SERVICE_NAMES,
            channels_steps=CHANNEL_STEPS,
            owner_id="+12345952496",
            status=SessionStatus.ACTIVE,
            first_channel=INPUT_CHANNEL,
            last_channel=OUTPUT_CHANNEL,
            timeout=3000,
            created_at=time.time(),
        )
        await redis_client.hset(ACTIVE_SESSIONS_KEY, sid, status.to_json())

        txt_objects.append(

            TextFeatures(
                sid=sid,
                agent_type=AGENT_NAME,
                priority="high",
                created_at=time.time(),
                is_final=False,
                text=sentence
            )
        )
    await redis_client.hset(ACTIVE_SESSIONS_KEY, sid, status.to_json())

    # txt_objects = [

    #     TextFeatures(
    #         sid="test_sid_4",
    #         agent_type="call",
    #         priority="high",
    #         created_at=time.time(),
    #         is_final=False,
    #         text="I love music."
    #     ),
    #     TextFeatures(
    #         sid="test_sid_2",
    #         agent_type="call",
    #         priority="high",
    #         created_at=time.time(),
    #         is_final=False,
    #         text="He ate a small apple."
    #     )
    # ]
    tasks = []
    for i in txt_objects:
        channel = f"TTS:{i.priority}"

        logger.info(
            f"Publishing TTS request (sid={i.sid}, priority={i.priority}) to {channel}"
        )
        tasks.append(redis_client.lpush(channel, i.to_json()))

    await asyncio.gather(*tasks)
    logger.info(f"Published {len(txt_objects)} {channel} requests.")


async def listen_for_rag_results(redis_client, expected_count: int, timeout: int = 60):
    pubsub = redis_client.pubsub()
    await pubsub.subscribe(*OUTPUT_CHANNELS)

    results = []
    start_time = asyncio.get_event_loop().time()

    logger.info("Listening for TTS results...")
    async for message in pubsub.listen():
        if message["type"] != "message":
            continue

        channel = message["channel"].decode()
        logger.info(f"✅ Received TTS result on {channel}")
        results.append(message["data"])

        if len(results) >= expected_count:
            break

        if asyncio.get_event_loop().time() - start_time > timeout:
            logger.warning("Timeout reached while waiting for RAG results.")
            break

    await pubsub.unsubscribe(*OUTPUT_CHANNELS)
    return results


async def main():
    NUM_REQUESTS = 2
    TIMEOUT = 60

    redis_client = await redis.from_url(
        "redis://localhost:6379", decode_responses=False
    )

    # Start listener before publishing
    listener_task = asyncio.create_task(
        listen_for_rag_results(redis_client, NUM_REQUESTS, TIMEOUT)
    )

    # Publish simulated STT outputs
    await publish_stt_requests(redis_client, num_requests=NUM_REQUESTS)

    # Wait for results
    try:
        results = await asyncio.wait_for(listener_task, timeout=TIMEOUT)
        logger.info(f"✅ Received {len(results)} / {NUM_REQUESTS} TTS results.")
    except asyncio.TimeoutError:
        logger.error("❌ Timeout: Not all TTS results received.")
        results = listener_task.result() if listener_task.done() else []

    # Cleanup: mark sessions as stopped
    sids = [f"test_sid_{i}" for i in range(NUM_REQUESTS)]
    for sid in sids:
        stop_status = SessionStatus(
            sid=sid, status="stop", created_at=None, timeout=0.0
        )
        await redis_client.hset(ACTIVE_SESSIONS_KEY, sid, stop_status.to_json())

    await redis_client.close()


if __name__ == "__main__":
    asyncio.run(main())
