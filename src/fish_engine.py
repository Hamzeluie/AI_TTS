"""
FishEngine - Fish TTS Engine implementation that inherits from BaseEngine.
This engine provides real-time text-to-speech synthesis using Fish TTS models.
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
import io
import time
import sys

# Assuming these imports are available from the Fish TTS module
sys.path.append(str(Path(__file__).resolve().parent.parent))
# from models.fish_speech.models.text2semantic.inference import (
#     main as generate_semantic_tokens
# )
from fish_speech.fish_speech.models.text2semantic.inference import (
    init_model as load_text2semantic_model,
    my_main as generate_semantic_tokens
)
from models.fish_speech.models.vqgan.inference import (
    main as vqgan_infer_main,
    infer_tts as vqgan_infer_tts_from_codes
)

from fish_speech.fish_speech.models.dac.inference import (
    load_model as load_vqgan_model,
)
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

class FishVoice:
    def __init__(self, name, features_path=None):
        self.name = name
        self.features_path = features_path

    def __repr__(self):
        return f"FishVoice({self.name})"


class FishEngine(BaseEngine):
    """
    Fish TTS Engine implementation that provides real-time text-to-speech synthesis.
    """

    def __init__(
        self,
        checkpoint_dir: str = "checkpoints",
        output_dir: str = "outputs",
        voices_path: Optional[str] = None,
        reference_prompt_tokens: Union[str, Path] = "default",
        device: Optional[str] = None,
        precision: str = "half",
        compile: bool = True,
        num_samples: int = 1,
        max_new_tokens: int = 1024,
        top_p: float = 0.7,
        repetition_penalty: float = 1.3,
        temperature: float = 0.3,
        chunk_length: int = 100,
        reference_prompt_text: str = "This is the reference prompt text, guiding the style of the cloned voice.",
        level: int = logging.WARNING,
        prepare_text_for_synthesis_callback=None,
    ):
        """
        Initialize the Fish TTS Engine.

        Args:
            checkpoint_dir (str): Directory containing model checkpoints
            output_dir (str): Directory for temporary output files
            voices_path (str, optional): Path to directory containing voice files
            voice (Union[str, Path]): Path to voice features file or voice name
            device (str, optional): Device to use ('cuda', 'cpu', or None for auto)
            precision (str): Model precision ('half' or 'full')
            compile (bool): Whether to compile the model for faster inference
            num_samples (int): Number of samples to generate
            max_new_tokens (int): Maximum number of tokens to generate
            top_p (float): Top-p sampling parameter
            repetition_penalty (float): Repetition penalty
            temperature (float): Sampling temperature
            chunk_length (int): Chunk length for iterative generation
            reference_prompt_text (str): Reference text for voice cloning
            level (int): Logging level
            prepare_text_for_synthesis_callback: Optional callback for text preprocessing
        """
        self._synthesize_lock = Lock()
        self.checkpoint_dir = Path(checkpoint_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.voices_path = Path(voices_path) if voices_path else None
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.precision = torch.half if precision == "half" else torch.float32
        self.compile = compile
        self.num_samples = num_samples
        self.max_new_tokens = max_new_tokens
        self.top_p = top_p
        self.repetition_penalty = repetition_penalty
        self.temperature = temperature
        self.chunk_length = chunk_length
        self.reference_prompt_text = reference_prompt_text
        self.level = level
        self.prepare_text_callback = prepare_text_for_synthesis_callback
        
        # Model paths
        self.vqgan_checkpoint = self.checkpoint_dir / "codec.pth"
        self.config_name = "modded_dac_vq"
        
        # Initialize voice
        self.reference_prompt_tokens = reference_prompt_tokens
        self.current_voice_features_path = None
        
        # Setup logging
        logging.basicConfig(format="FishEngine: %(message)s", level=level)

    def post_init(self):
        """Post-initialization setup called by BaseInitMeta."""
        self.engine_name = "fish"
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
        
        # Get available voices
        self.voices_list = self.get_voices()
        
        # Start synthesis process
        self.synthesize_process = mp.Process(
            target=FishEngine._synthesize_worker,
            args=(
                self.output_queue,
                child_synthesize_pipe,
                self.stop_synthesis_event,
                self.checkpoint_dir,
                self.output_dir,
                self.reference_prompt_tokens,
                self.device,
                self.precision,
                self.compile,
                self.num_samples,
                self.max_new_tokens,
                self.top_p,
                self.repetition_penalty,
                self.temperature,
                self.chunk_length,
                self.reference_prompt_text,
                self.main_synthesize_ready_event,
                self.level,
                self.voices_path,
                self.vqgan_checkpoint,
                self.config_name,
            ),
        )
        self.synthesize_process.start()
        
        logging.debug("Waiting for Fish TTS model initialization")
        # Add timeout to prevent infinite waiting
        if not self.main_synthesize_ready_event.wait(timeout=60):
            logging.error("Fish TTS model initialization timeout")
            raise RuntimeError("Failed to initialize Fish TTS model within timeout period")
        logging.info("Fish TTS model ready")

    @staticmethod
    def _synthesize_worker(
        output_queue,
        conn,
        stop_event,
        checkpoint_dir,
        output_dir,
        reference_prompt_tokens,
        device,
        precision,
        compile,
        num_samples,
        max_new_tokens,
        top_p,
        repetition_penalty,
        temperature,
        chunk_length,
        reference_prompt_text,
        ready_event,
        loglevel,
        voices_path,
        vqgan_checkpoint,
        config_name,
    ):
        """Worker process for Fish TTS synthesis."""
        # Redirect output to queue
        sys.stdout = QueueWriter(output_queue)
        sys.stderr = QueueWriter(output_queue)
        
        logging.basicConfig(format="FishEngine Worker: %(message)s", level=loglevel)
        logging.info("Starting Fish TTS synthesis worker")
        
        vqgan_model = None
        text2semantic_model = None
        
        try:
            # Verify checkpoint directory exists
            checkpoint_dir = Path(checkpoint_dir)
            if not checkpoint_dir.exists():
                raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")
            
            # Verify VQGAN checkpoint exists
            vqgan_checkpoint = Path(vqgan_checkpoint)
            if not vqgan_checkpoint.exists():
                raise FileNotFoundError(f"VQGAN checkpoint not found: {vqgan_checkpoint}")
            
            # Initialize models
            logging.info(f"Loading VQGAN model from {vqgan_checkpoint}...")
            vqgan_model = load_vqgan_model(
                config_name=config_name,
                checkpoint_path=str(vqgan_checkpoint),
                device=device
            )
            logging.info("VQGAN model loaded successfully")
            
            logging.info(f"Loading Text2Semantic model from {checkpoint_dir}...")
            
            text2semantic_model, decode_one_token_func = load_text2semantic_model(
                checkpoint_path=str(checkpoint_dir),
                device=device,
                precision=precision,
                compile=compile
            )
            logging.info("Text2Semantic model loaded successfully")
            
            # Setup model caches
            if device == "cuda":
                torch.cuda.synchronize()
            
            with torch.device(device):
                text2semantic_model.setup_caches(
                    max_batch_size=1,
                    max_seq_len=text2semantic_model.config.max_seq_len,
                    dtype=next(text2semantic_model.parameters()).dtype,
                )
            
            # Load initial voice
            cloned_voice_features_path = None
            if reference_prompt_tokens and reference_prompt_tokens != "default":
                try:
                    voice_path = Path(reference_prompt_tokens)
                    if voices_path and not voice_path.is_absolute():
                        voice_path = Path(voices_path) / voice_path
                    
                    if voice_path.exists():
                        if voice_path.suffix == ".npy":
                            cloned_voice_features_path = voice_path
                            logging.info(f"Using voice features from: {cloned_voice_features_path}")
                        elif voice_path.suffix == ".wav":
                            # Convert WAV to features
                            output_dir = Path(output_dir)
                            cloned_voice_features_path = output_dir / f"{voice_path.stem}_features.npy"
                            if not cloned_voice_features_path.exists():
                                logging.info(f"Extracting features from {voice_path}")
                                vqgan_infer_main(
                                    input_path=voice_path,
                                    output_path=cloned_voice_features_path.with_suffix(".wav"),
                                    checkpoint_path=vqgan_checkpoint,
                                    config_name=config_name,
                                    device=device,
                                    model=vqgan_model
                                )
                                logging.info(f"Voice features extracted to: {cloned_voice_features_path}")
                    else:
                        logging.warning(f"Voice file not found: {voice_path}")
                except Exception as e:
                    logging.error(f"Error loading initial voice: {e}")
                    logging.error(traceback.format_exc())
            
            # Signal ready
            ready_event.set()
            logging.info("Fish TTS synthesis worker initialized successfully")
            
            # Main processing loop
            while True:
                try:
                    if conn.poll(0.001):
                        try:
                            message = conn.recv()
                        except Exception as e:
                            logging.error(f"Error receiving message: {e}")
                            time.sleep(1)
                            continue
                    else:
                        time.sleep(0.001)
                        continue
                    
                    command = message["command"]
                    data = message["data"]
                    
                    if command == "synthesize":
                        stop_event.clear()
                        text = data["text"]
                        
                        try:
                            logging.debug(f"Synthesizing text: {text[:50]}...")
                            
                            # Create temp directory for this synthesis
                            temp_dir = Path(output_dir) / "temp"
                            temp_dir.mkdir(parents=True, exist_ok=True)
                            
                            # Generate semantic tokens
                            
                            prompt_tokens = [str(cloned_voice_features_path)] if cloned_voice_features_path else []
                            prompt_text = [reference_prompt_text] if cloned_voice_features_path else []
                            
                            logging.debug("Generating semantic tokens...")
                            semantic_tokens = generate_semantic_tokens(
                                text=text,
                                prompt_text=prompt_text,
                                prompt_tokens=prompt_tokens,
                                checkpoint_path=str(checkpoint_dir),
                                half=(precision == torch.half),
                                device=device,
                                num_samples=1,
                                max_new_tokens=max_new_tokens,
                                top_p=top_p,
                                repetition_penalty=repetition_penalty,
                                temperature=temperature,
                                compile=False,  # Already compiled
                                seed=42,
                                iterative_prompt=True,
                                chunk_length=chunk_length,
                                output_dir=str(temp_dir),
                                model=text2semantic_model,
                                decode_one_token=decode_one_token_func,
                            )
                            logging.debug("Semantic tokens generated")
                            
                            if stop_event.is_set():
                                logging.info("Synthesis stopped during token generation")
                                conn.send(("finished", ""))
                                continue
                            
                            # Convert tokens to audio
                            if isinstance(semantic_tokens, torch.Tensor):
                                tokens_tensor = semantic_tokens.to(device).long()
                            else:
                                # Load from file if path
                                tokens_tensor = torch.from_numpy(np.load(semantic_tokens)).to(device).long()
                            
                            logging.debug("Converting tokens to audio...")
                            audio_data, audio_duration = vqgan_infer_tts_from_codes(
                                npy_tts_voice=tokens_tensor,
                                device=device,
                                model=vqgan_model
                            )
                            logging.debug(f"Audio generated, duration: {audio_duration}s")
                            
                            if stop_event.is_set():
                                logging.info("Synthesis stopped during audio generation")
                                conn.send(("finished", ""))
                                continue
                            
                            # Convert to float32 PCM for streaming
                            audio_array = audio_data[0, 0].float().detach().cpu().numpy()
                            
                            # Check if we need to resample
                            # Fish TTS outputs at 44100 Hz, but BaseEngine expects 24000 Hz
                            try:
                                import scipy.signal
                                logging.debug("Resampling audio from 44100 to 24000 Hz")
                                audio_array = scipy.signal.resample(
                                    audio_array, 
                                    int(len(audio_array) * 24000 / 44100)
                                )
                            except ImportError:
                                logging.warning("scipy not available for resampling, sending at original sample rate")
                                # Note: This might cause issues if BaseEngine expects 24000 Hz
                            
                            # Send audio in chunks
                            chunk_size = 24000 // 4  # 0.25 second chunks
                            total_chunks = (len(audio_array) + chunk_size - 1) // chunk_size
                            logging.debug(f"Sending {total_chunks} audio chunks")
                            
                            for i in range(0, len(audio_array), chunk_size):
                                if stop_event.is_set():
                                    break
                                chunk = audio_array[i:i + chunk_size].astype(np.float32)
                                conn.send(("success", chunk.tobytes()))
                            
                            conn.send(("finished", ""))
                            logging.debug("Audio synthesis completed successfully")
                            
                            del semantic_tokens
                            del tokens_tensor
                            del audio_data
                            del audio_array
                            torch.cuda.empty_cache()
                                
                        except Exception as e:
                            logging.error(f"Synthesis error: {e}")
                            logging.error(traceback.format_exc())
                            conn.send(("error", str(e)))
                    
                    elif command == "update_voice":
                        new_voice_path = data["voice_path"]
                        try:
                            voice_path = Path(new_voice_path)
                            if voices_path and not voice_path.is_absolute():
                                voice_path = Path(voices_path) / voice_path
                            
                            if not voice_path.exists():
                                raise FileNotFoundError(f"Voice file not found: {voice_path}")
                            
                            if voice_path.suffix == ".npy":
                                cloned_voice_features_path = voice_path
                            elif voice_path.suffix == ".wav":
                                # Extract features
                                output_dir_path = Path(output_dir)
                                cloned_voice_features_path = output_dir_path / f"{voice_path.stem}_features.npy"
                                logging.info(f"Extracting features from {voice_path}")
                                vqgan_infer_main(
                                    input_path=voice_path,
                                    output_path=cloned_voice_features_path.with_suffix(".wav"),
                                    checkpoint_path=vqgan_checkpoint,
                                    config_name=config_name,
                                    device=device,
                                    model=vqgan_model
                                )
                            else:
                                raise ValueError(f"Unsupported voice file format: {voice_path.suffix}")
                            
                            conn.send(("success", "Voice updated successfully"))
                            logging.info(f"Voice updated to: {voice_path}")
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
            if vqgan_model is not None:
                del vqgan_model
            if text2semantic_model is not None:
                del text2semantic_model
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
                    if self.parent_synthesize_pipe.poll(30):  # 30 second timeout
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
        return pyaudio.paFloat32, 1, 24000

    def get_voices(self) -> List[FishVoice]:
        """
        Get list of available voices.
        
        Returns:
            List[FishVoice]: List of available voices
        """
        voices = []
        
        if self.voices_path and self.voices_path.exists():
            # Add voices from directory
            for file in self.voices_path.iterdir():
                if file.suffix in [".wav", ".npy"]:
                    voices.append(FishVoice(file.stem, file))
        
        # Add default voice
        voices.append(FishVoice("default", None))
        
        return voices

    def set_voice(self, voice: Union[str, Path, FishVoice]):
        """
        Set the voice for synthesis.
        
        Args:
            voice: Voice name, path, or FishVoice object
        """
        if isinstance(voice, FishVoice):
            voice_path = voice.features_path or voice.name
        elif isinstance(voice, (str, Path)):
            voice_path = voice
        else:
            raise ValueError(f"Invalid voice type: {type(voice)}")
        
        try:
            self.send_command("update_voice", {"voice_path": str(voice_path)})
            
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
        """Set voice parameters (not used in Fish TTS)."""
        pass

    def shutdown(self):
        """Shutdown the engine and cleanup resources."""
        logging.info("Shutting down FishEngine")
        
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
            
            logging.info("FishEngine shutdown complete")
        except Exception as e:
            logging.error(f"Error during shutdown: {e}")
            # Force terminate
            try:
                self.synthesize_process.terminate()
            except:
                pass