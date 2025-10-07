import asyncio
import logging
import os
import wave

import websockets
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

SERVER_HOST = os.getenv("HOST", "0.0.0.0")
SERVER_PORT = int(os.getenv("TEST_PORT", 5005))
WEBSOCKET_URI = f"ws://{SERVER_HOST}:{SERVER_PORT}"

SAMPLE_RATE = 8000
SAMPLE_WIDTH = 2
CHANNELS = 1
OUTPUT_FILE = "./received_audio_real.wav"

TEST_TEXT = "Hello world this is a test of the TTS WebSocket server."


class TestTTS:
    def __init__(self, uri, text):
        self.uri = uri
        self.text = text
        self.audio_chunks = []

    async def run(self):
        logger.info("Connecting to WebSocket server...")
        try:
            async with websockets.connect(self.uri, ping_interval=None) as ws:
                logger.info("Connected. Sending text...")
                await ws.send(self.text)

                logger.info("Receiving audio (with inactivity timeout)...")
                while True:
                    try:
                        # Wait up to 0.3 seconds for next message
                        message = await asyncio.wait_for(ws.recv(), timeout=0.3)
                        if isinstance(message, bytes):
                            logger.info(f"Received audio chunk: {len(message)} bytes")
                            self.audio_chunks.append(message)
                        elif isinstance(message, str):
                            if message in ("AUDIO_COMPLETE", "END_OF_AUDIO"):
                                logger.info("Received explicit completion signal.")
                                break
                            else:
                                logger.debug(f"Ignoring text message: {message}")
                    except asyncio.TimeoutError:
                        logger.info(
                            "No data received for 300ms — assuming end of audio stream."
                        )
                        break
                    except websockets.exceptions.ConnectionClosed:
                        logger.info("Server closed connection.")
                        break

            await self.save_audio()

        except Exception as e:
            logger.error(f"Error during TTS test: {e}", exc_info=True)
            raise

    async def save_audio(self):
        if not self.audio_chunks:
            logger.warning("No audio data received.")
            return

        total_bytes = sum(len(c) for c in self.audio_chunks)
        logger.info(
            f"Saving {len(self.audio_chunks)} chunks ({total_bytes} bytes) to {OUTPUT_FILE}"
        )

        with wave.open(OUTPUT_FILE, "wb") as wf:
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(SAMPLE_WIDTH)
            wf.setframerate(SAMPLE_RATE)
            for chunk in self.audio_chunks:
                wf.writeframes(chunk)

        logger.info(f"✅ Audio saved to {OUTPUT_FILE}")


async def main():
    logger.info("Starting TTS test...")
    client = TestTTS(WEBSOCKET_URI, TEST_TEXT)
    await client.run()
    logger.info("Test finished.")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Interrupted by user.")
    except Exception as e:
        logger.error(f"Unhandled error: {e}", exc_info=True)
