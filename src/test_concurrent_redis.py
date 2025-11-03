#!/usr/bin/env python3
"""
Test: Send TTS requests word by word for a single session.
→ Input sentence is split into words.
→ Each word → one TTS request.
→ Saves one WAV per word.
"""

import asyncio
import logging
import os
import sys
import wave
from pathlib import Path

import numpy as np
import redis.asyncio as redis

# Add project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent_architect.datatype_abstraction import AudioFeatures, TextFeatures
from agent_architect.session_abstraction import AgentSessions, SessionStatus

# --------------------------------------------------------------------------- #
# Logging
# --------------------------------------------------------------------------- #
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
REDIS_URL = "redis://localhost:6379"
AGENT_TYPE = "call"
SERVICE_NAMES = ["VAD", "STT", "RAG", "TTS"]
CHANNEL_STEPS = {
    "VAD": ["input"],
    "STT": ["high", "low"],
    "RAG": ["high", "low"],
    "TTS": ["high", "low"],
}
OUTPUT_CHANNEL = f"{AGENT_TYPE}:output"
ACTIVE_SESSIONS_KEY = f"{AGENT_TYPE}:active_sessions"

TTS_SAMPLE_RATE = 24000
SESSION_TIMEOUT = 30000
OUTPUT_DIR = Path("/home/ubuntu/borhan/whole_pipeline/vexu/AI_TTS/src/outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

# --------------------------------------------------------------------------- #
# Input sentence (will be split into words)
# --------------------------------------------------------------------------- #
INPUT_SENTENCE = "Hello Harvey how can I assist you today"
SESSION_ID = "tts_test_0"

# Split into words (preserves order, no punctuation stripping unless desired)
WORDS = INPUT_SENTENCE.split()  # e.g., ['Hello', 'Harvey', 'how', ...]


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
async def create_session(redis_client, sid: str):
    session = AgentSessions(
        sid=sid,
        agent_type=AGENT_TYPE,
        agent_id="TEST_AGENT",
        service_names=SERVICE_NAMES,
        channels_steps=CHANNEL_STEPS,
        owner_id="+1234567890",
        status=SessionStatus.ACTIVE,
        timeout=SESSION_TIMEOUT,
        first_channel="VAD:input",
        last_channel=OUTPUT_CHANNEL,
        created_at=None,
    )
    await redis_client.hset(ACTIVE_SESSIONS_KEY, sid, session.to_json())
    logger.info(f"Session created: {sid}")


async def publish_tts_request(redis_client, sid: str, word: str, priority: str):
    req = TextFeatures(
        sid=sid,
        agent_type=AGENT_TYPE,
        text=word,
        priority=priority,
        created_at=None,
        is_final=True,
    )
    channel = f"TTS:{priority}"
    await redis_client.lpush(channel, req.to_json())
    logger.info(f"Pushed | sid={sid} | pri={priority} | word='{word}'")


async def collect_all_audio_from_output(
    redis_client, expected_count: int, timeout: int = 180
) -> list:
    results = []
    start_time = asyncio.get_event_loop().time()
    logger.info(f"Waiting for {expected_count} results on {OUTPUT_CHANNEL}...")

    while len(results) < expected_count:
        elapsed = asyncio.get_event_loop().time() - start_time
        if elapsed > timeout:
            logger.warning(
                f"Timeout after {timeout}s. Got {len(results)}/{expected_count}"
            )
            break

        result = await redis_client.brpop(OUTPUT_CHANNEL, timeout=5)
        if result is None:
            continue

        _, data = result
        try:
            audio_feat = AudioFeatures.from_json(data)
            results.append(audio_feat)
            duration = len(audio_feat.audio) / TTS_SAMPLE_RATE
            logger.info(f"Received | sid={audio_feat.sid} | {duration:.2f}s")
        except Exception as e:
            logger.error(f"Parse error: {e}")

    return results


def save_wav(pcm_bytes: bytes, sample_rate: int, output_path: Path):
    audio_np = np.frombuffer(pcm_bytes, dtype=np.float32)
    audio_np = np.clip(audio_np, -1.0, 1.0)
    audio_int16 = (audio_np * 32767).astype(np.int16)

    with wave.open(str(output_path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio_int16.tobytes())

    logger.info(f"Saved {output_path.name}")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
async def main():
    redis_client = await redis.from_url(REDIS_URL, decode_responses=False)

    # 1. Create session
    await create_session(redis_client, SESSION_ID)

    # 2. Publish one TTS request per word
    total_words = len(WORDS)
    logger.info(f"Split input into {total_words} words: {WORDS}")

    publish_tasks = []
    for idx, word in enumerate(WORDS):
        priority = "high" if idx % 2 == 0 else "low"
        publish_tasks.append(
            publish_tts_request(redis_client, SESSION_ID, word, priority)
        )

    await asyncio.gather(*publish_tasks)
    logger.info(f"Published {total_words} word-level TTS requests")

    # 3. Collect all audio responses
    all_results = await collect_all_audio_from_output(
        redis_client, total_words, timeout=300
    )

    # 4. Save each result as a WAV file
    for idx, audio_feat in enumerate(all_results):
        if audio_feat.sid != SESSION_ID:
            logger.warning(f"Unexpected session: {audio_feat.sid}")
            continue
        output_path = OUTPUT_DIR / f"word_{SESSION_ID}_{idx:03d}_{WORDS[idx]}.wav"
        save_wav(audio_feat.audio, TTS_SAMPLE_RATE, output_path)

    # 5. Cleanup
    await redis_client.hdel(ACTIVE_SESSIONS_KEY, SESSION_ID)
    await redis_client.aclose()
    logger.info("Test completed.")


if __name__ == "__main__":
    asyncio.run(main())
