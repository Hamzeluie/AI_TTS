import os
import time
import sys
from pathlib import Path
from SmartAITool.core import cprint

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from models.fish_speech.models.text2semantic.inference import main as generate
from pathlib import Path
from models.fish_speech.models.vqgan.inference import main as infer
from models.fish_speech.models.vqgan.inference import load_model
from models.fish_speech.models.text2semantic.inference import load_model as load_text2semantic_model 
from models.fish_speech.models.vqgan.inference import infer_tts
import torch
import soundfile as sf


class VoiceService:
    def __init__(self):
        self._output_dir = "../outputs/"
        os.makedirs(self._output_dir, exist_ok=True)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.precision = torch.half
        self.checkpoint_path = Path("/home/ubuntu/ai_service/checkpoints/")
        self.vqgan_checkpoint_path = "/home/ubuntu/ai_service/checkpoints/firefly-gan-vq-fsq-8x1024-21hz-generator.pth"
        self.config_name = "firefly_gan_vq"
        
        # Initialize models to None - will be loaded on first use or warm-up
        self.model = None
        self.text2semantic_model = None
        self.decode_one_token = None
        self.voice_text = "accessing alarm and interface settings in this window you can set up your customized greeting and alarm preferences the world needs your expertise or at least your presence launching a series of displays to help guide you",
        # Perform warm-up on initialization
        self._initialize_models()
        self._warm_up_models()

    def _initialize_models(self):
        """Initialize the models if they haven't been loaded yet."""
        if self.model is None:
            cprint("Loading VQGAN model...", 'magenta')
            load_start = time.time()
            self.model = load_model(
                config_name=self.config_name,
                checkpoint_path=self.vqgan_checkpoint_path,
                device=self.device
            )
            cprint(f"VQGAN model loaded in {time.time() - load_start:.2f} seconds")
        
        if self.text2semantic_model is None:
            cprint("Loading Text2Semantic model...", 'magenta')
            load_start = time.time()
            self.text2semantic_model, self.decode_one_token = load_text2semantic_model(
                checkpoint_path=self.checkpoint_path,
                device=self.device,
                precision=self.precision,
                compile=True
            )
            cprint(f"Text2Semantic model loaded in {time.time() - load_start:.2f} seconds")

            if torch.cuda.is_available():
                torch.cuda.synchronize()

            with torch.device(self.device):
                self.text2semantic_model.setup_caches(
                    max_batch_size=1,
                    max_seq_len=self.text2semantic_model.config.max_seq_len,
                    dtype=next(self.text2semantic_model.parameters()).dtype,
                )

    def _warm_up_models(self):
        """Perform a warm-up inference pass with minimal text to initialize all optimizations."""
        cprint("Warming up models with a test inference...", 'magenta')
        warm_up_start = time.time()
        
        # Ensure models are loaded
        self._initialize_models()
        
        # Create a small clone file for warm-up if needed
        clone_path = Path(self._output_dir + "clone.wav")
        clone_npy_path = Path(self._output_dir + "clone.npy")
        
        if not clone_npy_path.exists():
            infer(
                input_path=Path("../../input/jarvis.wav"),  #FIXME
                output_path=clone_path,
                checkpoint_path=self.vqgan_checkpoint_path,
                config_name=self.config_name, 
                device=self.device, 
                model=self.model
            )
        
        # Run a minimal text-to-semantic generation
        warm_up_text = "This is a warm-up test."
        generate(
            text=warm_up_text,
            prompt_text=["Brief test prompt"],
            prompt_tokens=[clone_npy_path] if clone_npy_path.exists() else None,
            checkpoint_path=self.checkpoint_path,
            half=True,
            device=self.device,
            num_samples=1,  # Use minimum samples for warm-up
            max_new_tokens=50,  # Short sequence for warm-up
            top_p=0.7,
            repetition_penalty=1.2,
            temperature=0.3,
            compile=True,
            seed=42,
            iterative_prompt=True,
            chunk_length=100,
            output_dir=self._output_dir,
            model=self.text2semantic_model,
            decode_one_token=self.decode_one_token,
        )
        
        # Warm up the VQGAN inference
        if os.path.exists(f"{self._output_dir}/codes_1.npy"):
            infer(
                input_path=Path(f"{self._output_dir}/codes_1.npy"), 
                output_path=Path(self._output_dir + "warmup.wav"),
                checkpoint_path=self.vqgan_checkpoint_path,
                config_name=self.config_name, 
                device=self.device, 
                model=self.model
            )
        
        cprint(f"Model warm-up completed in {time.time() - warm_up_start:.2f} seconds")

    def clone_voice(self, voice_path=None, voice_text=None, output_folder=None):
        if voice_text is not None:
            self.voice_text = voice_text
        """Clone a voice from the provided audio file."""
        start_time = time.time()
        # Ensure models are loaded (will skip if already loaded)
        self._initialize_models()
        
        # Start execution timer after ensuring models are loaded
        cprint("Cloning voice...", 'magenta')
        
        # Use default voice path if none is provided
        if voice_path is None:
            voice_path = "../../input/jarvis.wav"  #FIXME
        
        # Use default output folder if none is provided
        if output_folder is None:
            output_folder = self._output_dir
        
        # Ensure the output folder exists
        os.makedirs(output_folder, exist_ok=True)
        
        # Define the output file path
        output_path = Path(output_folder) / "clone_voice_feature.wav"
        
        infer(
            input_path=Path(voice_path), 
            output_path=output_path,
            checkpoint_path=self.vqgan_checkpoint_path,
            config_name=self.config_name, 
            device=self.device, 
            model=self.model
        )
        
        cprint(f"Voice cloning completed in {time.time() - start_time:.2f} seconds")
        return output_path
    
    #NOTE main function 
    def fishspeech(self, text, clone_npy_path=None):

        start_time = time.time()
        
        # Ensure models are loaded (will skip if already loaded)
        self._initialize_models()

        # Clone voice if clone_npy_path is None
        if clone_npy_path is None:
            cprint("Clone path not provided. Using default clone voice feature path...", 'magenta')
            clone_voice_path = self.clone_voice()
            clone_npy_path = str(clone_voice_path.with_suffix('.npy'))
            cprint(f"Default clone voice feature saved at {clone_npy_path}", 'yellow')
        else:
            cprint(f"Using provided clone voice feature path: {clone_npy_path}", 'yellow')
        #--------------------------------generate audio---------------------------------#
        execution_start_time = time.time()
        generate_start = time.time()
        npy_tts_voice = generate(
            text=text,
            prompt_text = list(self.voice_text),
            prompt_tokens=[Path(clone_npy_path)],
            checkpoint_path=self.checkpoint_path,
            half=True,
            device=self.device,
            num_samples=2,
            max_new_tokens=1024,
            top_p=0.7,
            repetition_penalty=1.2,
            temperature=0.3,
            compile=True,
            seed=42,
            iterative_prompt=True,
            chunk_length=100,
            output_dir=self._output_dir,
            model=self.text2semantic_model,
            decode_one_token=self.decode_one_token,
        )
        cprint(f"Audio generation time: {time.time() - generate_start:.2f} seconds")


        #----------------------------------infer audio---------------------------------#
        infer_audio_start = time.time()
        audio, audio_duration = infer_tts(
            npy_tts_voice=npy_tts_voice, 
            device=self.device, 
            model=self.model
        )
        final_time = time.time()
        # cprint(f"Final audio inference time: {time.time() - infer_audio_start:.2f} seconds")
        execute_time = final_time - execution_start_time
        cprint(f"Execution time (excluding model loading): {execute_time:.2f} seconds")

        #-------------------save audio---------------------------------#
        output_audio_path = Path("../../outputs/result.wav")  # FIXME
        output_audio_folder = output_audio_path.parent
        output_audio_folder.mkdir(parents=True, exist_ok=True)  # Create folder if not exist
        sf.write(output_audio_path, audio[0, 0].float().detach().cpu().numpy(), 44100)
        cprint(f"audio duration is {audio_duration:.2f} seconds", 'yellow')
        cprint(f"Audio saved to {output_audio_path}", 'yellow')
        cprint(f"ratio speed is {audio_duration / execute_time:.2f}", 'green')
    #--------------------------------------------------------------------------------

    # def play(self, temp_audio_file):
    #     pygame.mixer.quit()
    #     pygame.mixer.init()
    #     pygame.mixer.music.load(temp_audio_file)
    #     pygame.mixer.music.stop()
    #     pygame.mixer.music.play()

    #     while pygame.mixer.music.get_busy():
    #         pygame.time.Clock().tick(10)

    #     pygame.mixer.music.stop()
    #     pygame.mixer.quit()

if __name__ == "__main__":

    vs = VoiceService()
    # Uncomment the following line to clone a voice if needed
    # vs.clone_voice()

    text = "I'm borhan. I'm curious robot. How can I help you today? I'm borhan. I'm curious robot. How can I help you today? I'm borhan. I'm curious robot. How can I help you today?"
    vs.fishspeech(
        text=text,
        clone_npy_path="/root/m15kh/ai_service/input/clone_voice_feature.npy",
    )