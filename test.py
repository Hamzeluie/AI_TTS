import asyncio
import base64
import logging
import os
import uuid
import wave
from pathlib import Path

import websockets
from dotenv import load_dotenv

load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Server configuration - adjust these based on your setup
SERVER_HOST = os.getenv("HOST", "0.0.0.0")
SERVER_PORT = int(os.getenv("PORT", 8000))
WEBSOCKET_URI = f"ws://{SERVER_HOST}:{SERVER_PORT}"

# Audio configuration for saving received audio
TARGET_SAMPLE_RATE = 8000
TARGET_SAMPLE_WIDTH = 2  # 16-bit
TARGET_CHANNELS = 1  # Assuming mono
OUTPUT_WAV_FILE = "./received_audio_real.wav"

# Test text to send - split into words/tokens
TEST_TEXT = "Hello world this is a test of the TTS WebSocket server."
TOKENS = TEST_TEXT.split()  # Send word by word for streaming


class TestTTS:
    def __init__(
        self, uri, input_text="Hello world this is a test of the TTS WebSocket server."
    ):
        self.uri = uri
        self.input_text = input_text
        self.connected = False
        self.websocket = None
        self.formatted_audio_responses = []  # To collect audio deltas
        self.tts_responses = []  # To collect metadata/events
        self.current_item_id = str(uuid.uuid4())  # Unique ID for this request

    async def connect(self):
        logger.info(f"Connecting to WebSocket server at {self.uri}")
        try:
            self.websocket = await websockets.connect(self.uri, ping_interval=None)
            self.connected = True
            logger.info("Connected successfully.")
        except Exception as e:
            logger.error(f"Failed to connect: {e}")
            raise

    async def send(self):
        if not self.connected:
            raise Exception("TTS WebSocket not connected")
        try:
            logger.info(f"Sending full text: {self.input_text}")
            await self.websocket.send(self.input_text)
            # DO NOT send END_OF_TEXT — let server decide when to finish
        except Exception as e:
            logger.error(f"Error sending text to TTS: {e}")
            raise

    async def message_receiver(self):
        """Listens for audio from the TTS server."""
        logger.info("Starting to receive messages...")
        try:
            audio_complete = False
            while self.connected and not audio_complete:
                try:
                    message = await self.websocket.recv()
                    if isinstance(message, bytes):
                        # Process audio chunk
                        logger.info(f"Received audio chunk of {len(message)} bytes")
                        audio_chunk_b64 = base64.b64encode(message).decode("utf-8")
                        tts_response = {
                            "type": "response.audio.delta",
                            "delta": audio_chunk_b64,
                            "item_id": self.current_item_id,
                        }
                        self.formatted_audio_responses.append(tts_response)
                        break

                    elif isinstance(message, str) and message in (
                        "AUDIO_COMPLETE",
                        "END_OF_AUDIO",
                    ):
                        # TTS signals all audio for the current text is done
                        logger.info(f"Received completion signal: {message}")
                        audio_complete = True
                        self.tts_responses.append(
                            {
                                "type": "response.done",
                                "item_id": self.current_item_id,
                            }
                        )
                        break

                    else:
                        logger.warning(f"Unexpected message type or content: {message}")

                except asyncio.TimeoutError:
                    continue

            # Close if still connected
            if self.websocket and self.connected:
                await self.websocket.close()
            self.connected = False
            logger.info("WebSocket connection closed gracefully.")
            return True

        except websockets.exceptions.ConnectionClosedOK:
            logger.info(
                "Server closed connection gracefully (code 1000). Audio stream likely complete."
            )
            self.connected = False
            return True

        except Exception as e:
            logger.error(f"An error occurred in TTS receiver: {e}")
            if self.websocket and self.connected:
                await self.websocket.close()
            self.connected = False
            return False

    async def save_audio_to_wav(self):
        """Saves all received audio chunks into a WAV file."""
        if not self.formatted_audio_responses:
            logger.warning("No audio received to save.")
            return

        logger.info(f"Saving audio to {OUTPUT_WAV_FILE}...")

        with wave.open(OUTPUT_WAV_FILE, "wb") as wf:
            wf.setnchannels(TARGET_CHANNELS)
            wf.setsampwidth(TARGET_SAMPLE_WIDTH)
            wf.setframerate(TARGET_SAMPLE_RATE)

            for resp in self.formatted_audio_responses:
                audio_data = base64.b64decode(resp["delta"])
                wf.writeframes(audio_data)

        logger.info(f"Audio saved successfully to {OUTPUT_WAV_FILE}")

    async def run(self):
        """Main method to run the full test: connect, send, receive, save."""
        try:
            await self.connect()
            await self.send()
            success = await self.message_receiver()
            if success:
                self.input_text = "END_OF_TEXT"
                await self.connect()
                await self.send()
                success = await self.message_receiver()
            if success:
                logger.info("✅ Test completed successfully.")

        except Exception as e:
            logger.error(f"Test failed with exception: {e}")


async def main():
    """Entry point for the test script."""
    logger.info("Starting WebSocket TTS client test...")

    tts_test = TestTTS(uri=WEBSOCKET_URI, input_text=TEST_TEXT)

    await tts_test.run()

    logger.info("Test script finished.")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Test interrupted by user.")
    except Exception as e:
        logger.error(f"Unhandled exception: {e}")
