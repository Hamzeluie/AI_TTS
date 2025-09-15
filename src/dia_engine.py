"""
DiaEngine - Dia TTS Engine implementation that inherits from BaseEngine.
This engine provides real-time text-to-speech synthesis using Dia models.
"""
import torch.multiprocessing as mp
from threading import Lock, Thread
from typing import Union, List, Optional
from pathlib import Path
import numpy as np
import traceback
import pyaudio
import logging
import torch
import queue
import time
import sys
import os
import io

# Import Dia model
from dia.model import Dia

from RealtimeTTS import BaseEngine
from RealtimeTTS import SafePipe


class QueueWriter(io.TextIOBase):
    """
    Custom file-like object to write text to a multiprocessing queue.
    """

    def __init__(self, queue):
        super().__init__()
        self.queue = queue

    def write(self, msg):
        """
        Write the message to the queue.

        Args:
            msg (str): The message to write.
        """
        if msg.strip():
            self.queue.put(msg)


class DiaVoice:
    def __init__(self, name, voice_id=None, description=None):
        self.name = name
        self.voice_id = voice_id
        self.description = description

    def __repr__(self):
        return f"DiaVoice({self.name})"


class DiaEngine(BaseEngine):
    """
    Dia TTS Engine implementation that provides real-time text-to-speech synthesis.
    """

    def __init__(
        self,
        model_name: str = "nari-labs/Dia-1.6B",
        device: Optional[str] = None,
        compute_dtype: str = "float16",
        use_torch_compile: bool = True,
        voice: Optional[str] = None,
        level: int = logging.WARNING,
        prepare_text_for_synthesis_callback=None,
        chunk_duration_ms: int = 250,  # Duration of each audio chunk in milliseconds
        output_sample_rate: int = 24000,  # Output sample rate for BaseEngine compatibility
    ):
        """
        Initialize the Dia TTS Engine.

        Args:
            model_name (str): Name of the Dia model to use
            device (str, optional): Device to use ('cuda', 'cpu', or None for auto)
            compute_dtype (str): Compute data type ('float16' or 'float32')
            use_torch_compile (bool): Whether to use torch.compile for faster inference
            voice (str, optional): Voice to use (if Dia supports multiple voices)
            level (int): Logging level
            prepare_text_for_synthesis_callback: Optional callback for text preprocessing
            chunk_duration_ms (int): Duration of each audio chunk in milliseconds
            output_sample_rate (int): Output sample rate (24000 for BaseEngine compatibility)
        """
        self._synthesize_lock = Lock()
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.compute_dtype = compute_dtype
        self.use_torch_compile = use_torch_compile
        self.current_voice = voice
        self.level = level
        self.prepare_text_callback = prepare_text_for_synthesis_callback
        self.chunk_duration_ms = chunk_duration_ms
        self.output_sample_rate = output_sample_rate
        
        # Setup logging
        logging.basicConfig(format="DiaEngine: %(message)s", level=level)

    def post_init(self):
        """Post-initialization setup called by BaseInitMeta."""
        self.engine_name = "dia"
        self.create_worker_process()

    def create_worker_process(self):
        """Create and start the worker process for synthesis."""
        self.output_queue = mp.Queue()
        
        # Start output worker thread
        def output_worker(q):
            while True:
                message = q.get()
                if message == "STOP":
                    break
                print(message)
        
        self.output_worker_thread = Thread(target=output_worker, args=(self.output_queue,))
        self.output_worker_thread.daemon = True
        self.output_worker_thread.start()
        
        # Create pipes and events
        self.main_synthesize_ready_event = mp.Event()
        self.parent_synthesize_pipe, child_synthesize_pipe = SafePipe()
        
        # Start synthesis process
        self.synthesize_process = mp.Process(
            target=DiaEngine._synthesize_worker,
            args=(
                self.output_queue,
                child_synthesize_pipe,
                self.stop_synthesis_event,
                self.model_name,
                self.device,
                self.compute_dtype,
                self.use_torch_compile,
                self.current_voice,
                self.main_synthesize_ready_event,
                self.level,
                self.chunk_duration_ms,
                self.output_sample_rate,
            ),
        )
        self.synthesize_process.start()
        
        logging.debug("Waiting for Dia model initialization")
        # Add timeout to prevent infinite waiting
        if not self.main_synthesize_ready_event.wait(timeout=120):  # Longer timeout for model loading
            logging.error("Dia model initialization timeout")
            raise RuntimeError("Failed to initialize Dia model within timeout period")
        logging.info("Dia model ready")

    @staticmethod
    def _synthesize_worker(
        output_queue,
        conn,
        stop_event,
        model_name,
        device,
        compute_dtype,
        use_torch_compile,
        initial_voice,
        ready_event,
        loglevel,
        chunk_duration_ms,
        output_sample_rate,
    ):
        """Worker process for Dia TTS synthesis."""
        # Redirect output to queue
        sys.stdout = QueueWriter(output_queue)
        sys.stderr = QueueWriter(output_queue)
        
        logging.basicConfig(format="DiaEngine Worker: %(message)s", level=loglevel)
        logging.info("Starting Dia TTS synthesis worker")
        
        model = None
        current_voice = initial_voice
        
        try:
            # Initialize Dia model
            logging.info(f"Loading Dia model: {model_name}")
            model = Dia.from_pretrained(
                model_name, 
                compute_dtype=compute_dtype, 
                device=device
            )
            logging.info("Dia model loaded successfully")
            
            # Signal ready
            ready_event.set()
            logging.info("Dia synthesis worker initialized successfully")
            
            # Main processing loop
            while True:
                try:
                    if conn.poll(0.01):
                        try:
                            message = conn.recv()
                        except Exception as e:
                            logging.error(f"Error receiving message: {e}")
                            time.sleep(1)
                            continue
                    else:
                        time.sleep(0.01)
                        continue
                    
                    command = message["command"]
                    data = message["data"]
                    
                    if command == "synthesize":
                        stop_event.clear()
                        text = data["text"]
                        
                        try:
                            logging.debug(f"Synthesizing text: {text[:50]}...")
                            
                            # Generate audio with Dia
                            start_time = time.time()
                            
                            # Generate audio
                            # Note: Based on the example, generate returns audio data
                            # We may need to pass voice parameters if Dia supports them
                            audio_output = model.generate(
                                text, 
                                use_torch_compile=use_torch_compile,
                                verbose=False  # Set to False to reduce output
                            )
                            
                            generation_time = time.time() - start_time
                            logging.debug(f"Audio generation took {generation_time:.2f} seconds")
                            
                            if stop_event.is_set():
                                logging.info("Synthesis stopped during generation")
                                conn.send(("finished", ""))
                                continue
                            
                            # audio_output is already a numpy array with float32 dtype
                            # Dia outputs at 44100 Hz sample rate
                            dia_sample_rate = 44100
                            audio_array = audio_output
                            
                            # Convert to mono if stereo
                            if len(audio_array.shape) > 1:
                                audio_array = np.mean(audio_array, axis=1)
                            
                            # Ensure float32
                            if audio_array.dtype != np.float32:
                                audio_array = audio_array.astype(np.float32)
                            
                            # Resample from 44100 to output_sample_rate (24000) if necessary
                            if dia_sample_rate != output_sample_rate:
                                try:
                                    import scipy.signal
                                    logging.debug(f"Resampling audio from {dia_sample_rate} to {output_sample_rate} Hz")
                                    audio_array = scipy.signal.resample(
                                        audio_array,
                                        int(len(audio_array) * output_sample_rate / dia_sample_rate)
                                    )
                                except ImportError:
                                    logging.warning("scipy not available for resampling, audio will be at 44100 Hz")
                                    # Note: This might cause issues if BaseEngine expects 24000 Hz
                            
                            # Normalize audio to [-1, 1] range if needed
                            max_val = np.max(np.abs(audio_array))
                            if max_val > 1.0:
                                audio_array = audio_array / max_val
                                logging.debug(f"Normalized audio (max value was {max_val})")
                            
                            # Send audio in chunks
                            chunk_size = int(output_sample_rate * chunk_duration_ms / 1000)
                            total_chunks = (len(audio_array) + chunk_size - 1) // chunk_size
                            logging.debug(f"Sending {total_chunks} audio chunks")
                            
                            for i in range(0, len(audio_array), chunk_size):
                                if stop_event.is_set():
                                    break
                                chunk = audio_array[i:i + chunk_size].astype(np.float32)
                                conn.send(("success", chunk.tobytes()))
                            
                            conn.send(("finished", ""))
                            logging.debug("Audio synthesis completed successfully")
                            
                        except Exception as e:
                            logging.error(f"Synthesis error: {e}")
                            logging.error(traceback.format_exc())
                            conn.send(("error", str(e)))
                    
                    elif command == "update_voice":
                        new_voice = data["voice"]
                        try:
                            # Update voice if Dia supports voice selection
                            # This is a placeholder - actual implementation depends on Dia's API
                            current_voice = new_voice
                            logging.info(f"Voice updated to: {new_voice}")
                            conn.send(("success", "Voice updated successfully"))
                        except Exception as e:
                            logging.error(f"Failed to update voice: {e}")
                            conn.send(("error", f"Failed to update voice: {e}"))
                    
                    elif command == "shutdown":
                        logging.info("Shutdown command received")
                        conn.send(("shutdown", "shutdown"))
                        break
                        
                except Exception as e:
                    logging.error(f"Error in main loop: {e}")
                    logging.error(traceback.format_exc())
                    # Don't exit the loop on errors, try to recover
                    time.sleep(0.1)
        
        except Exception as e:
            logging.error(f"Fatal worker process error: {e}")
            logging.error(traceback.format_exc())
            # Signal that initialization failed if we haven't set ready yet
            if not ready_event.is_set():
                ready_event.set()
            # Try to send error message before exiting
            try:
                conn.send(("error", f"Worker initialization failed: {str(e)}"))
            except:
                pass
        finally:
            # Cleanup
            logging.info("Cleaning up worker process")
            if model is not None:
                del model
            if device == "cuda":
                torch.cuda.empty_cache()
            # Restore stdout/stderr
            sys.stdout = sys.__stdout__
            sys.stderr = sys.__stderr__

    def synthesize(self, text: str) -> bool:
        """
        Synthesize text to audio.
        
        Args:
            text (str): Text to synthesize
            
        Returns:
            bool: True if synthesis was successful
        """
        super().synthesize(text)
        
        with self._synthesize_lock:
            # Check if worker process is still alive
            if not self.synthesize_process.is_alive():
                logging.error("Worker process is not alive, cannot synthesize")
                return False
            
            # Prepare text if callback provided
            if self.prepare_text_callback:
                text = self.prepare_text_callback(text)
            
            text = text.strip()
            if len(text) < 1:
                return False
            
            # Send synthesis command
            try:
                self.send_command("synthesize", {"text": text})
            except Exception as e:
                logging.error(f"Failed to send synthesis command: {e}")
                return False
            
            # Process response
            while True:
                try:
                    # Add timeout to prevent infinite waiting
                    if self.parent_synthesize_pipe.poll(60):  # 60 second timeout for Dia
                        response = self.parent_synthesize_pipe.recv()
                    else:
                        logging.error("Synthesis timeout: No response from worker")
                        return False
                    
                    # Check if pipe is closed or broken
                    if response is None:
                        logging.error("Synthesis failed: Pipe connection broken or closed")
                        return False
                    
                    status, result = response
                    if status == "finished":
                        return True
                    elif status == "error":
                        logging.error(f"Synthesis error: {result}")
                        return False
                    elif status == "shutdown":
                        return False
                    elif status == "success" and isinstance(result, bytes):
                        self.queue.put(result)
                    
                    if self.stop_synthesis_event.is_set():
                        return False
                        
                except EOFError:
                    logging.error("Pipe closed unexpectedly")
                    return False
                except Exception as e:
                    logging.error(f"Error during synthesis: {e}")
                    return False

    def send_command(self, command: str, data: dict):
        """Send command to worker process."""
        message = {"command": command, "data": data}
        try:
            self.parent_synthesize_pipe.send(message)
        except Exception as e:
            logging.error(f"Failed to send command '{command}': {e}")
            raise

    def get_stream_info(self):
        """
        Get audio stream configuration.
        
        Returns:
            tuple: (format, channels, sample_rate)
        """
        return pyaudio.paFloat32, 1, self.output_sample_rate

    def get_voices(self) -> List[DiaVoice]:
        """
        Get list of available voices.
        
        Returns:
            List[DiaVoice]: List of available voices
        """
        # This is a placeholder - actual implementation depends on what voices Dia supports
        voices = [
            DiaVoice("default", "default", "Default Dia voice"),
        ]
        
        # If Dia has predefined voices, add them here
        # Example:
        # voices.extend([
        #     DiaVoice("voice1", "voice1_id", "Description of voice 1"),
        #     DiaVoice("voice2", "voice2_id", "Description of voice 2"),
        # ])
        
        return voices

    def set_voice(self, voice: Union[str, DiaVoice]):
        """
        Set the voice for synthesis.
        
        Args:
            voice: Voice name or DiaVoice object
        """
        if isinstance(voice, DiaVoice):
            voice_id = voice.voice_id or voice.name
        else:
            voice_id = voice
        
        try:
            self.send_command("update_voice", {"voice": voice_id})
            
            # Wait for response with timeout
            if self.parent_synthesize_pipe.poll(10):  # 10 second timeout
                response = self.parent_synthesize_pipe.recv()
                if response is None:
                    logging.error("Failed to update voice: Pipe connection broken or closed")
                    return
                
                status, result = response
                
                if status == "success":
                    self.current_voice = voice
                    logging.info(f"Voice updated to: {voice}")
                else:
                    logging.error(f"Failed to update voice: {result}")
            else:
                logging.error("Voice update timeout")
        except Exception as e:
            logging.error(f"Error updating voice: {e}")

    def set_voice_parameters(self, **kwargs):
        """
        Set voice parameters.
        
        Args:
            **kwargs: Voice parameters (if supported by Dia)
        """
        # This is a placeholder - implement based on what parameters Dia supports
        # For example:
        # - speaking_rate
        # - pitch
        # - volume
        # etc.
        pass

    def shutdown(self):
        """Shutdown the engine and cleanup resources."""
        logging.info("Shutting down DiaEngine")
        
        try:
            # Send shutdown command
            self.send_command("shutdown", {})
            
            # Stop output worker
            self.output_queue.put("STOP")
            self.output_worker_thread.join(timeout=5)
            
            # Wait for acknowledgment with timeout
            try:
                if self.parent_synthesize_pipe.poll(5):  # 5 second timeout
                    status, _ = self.parent_synthesize_pipe.recv()
                    if status == "shutdown":
                        logging.info("Worker acknowledged shutdown")
            except EOFError:
                logging.warning("Worker pipe closed before acknowledgment")
            
            # Close pipe and terminate process
            self.parent_synthesize_pipe.close()
            
            # Give the process a chance to exit gracefully
            self.synthesize_process.join(timeout=5)
            
            # Force terminate if still alive
            if self.synthesize_process.is_alive():
                logging.warning("Worker process did not exit gracefully, terminating...")
                self.synthesize_process.terminate()
                self.synthesize_process.join()
            
            logging.info("DiaEngine shutdown complete")
        except Exception as e:
            logging.error(f"Error during shutdown: {e}")