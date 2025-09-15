import os
import time
import sys
from pathlib import Path
from SmartAITool.core import cprint # Assuming this is your custom print utility

# Adjust Python path to find ai_service modules
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from models.fish_speech.models.text2semantic.inference import main as generate_semantic_tokens
from models.fish_speech.models.vqgan.inference import main as vqgan_infer_main
from models.fish_speech.models.vqgan.inference import load_model as load_vqgan_model
from models.fish_speech.models.text2semantic.inference import load_model as load_text2semantic_model
from models.fish_speech.models.vqgan.inference import infer_tts as vqgan_infer_tts_from_codes
import torch
import soundfile as sf
import numpy as np # Ensure numpy is imported


class VoiceService:
    def __init__(self, reference_voice_input_path: str, reference_prompt_text: str):
        self._output_dir = Path("/home/ubuntu/ai_service/outputs/") 
        self._output_dir.mkdir(parents=True, exist_ok=True)
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.precision = torch.half 
        
        self.checkpoint_path = Path("/home/ubuntu/ai_service/checkpoints/")
        self.vqgan_checkpoint_path = self.checkpoint_path / "firefly-gan-vq-fsq-8x1024-21hz-generator.pth"
        self.config_name = "firefly_gan_vq"

        self.vqgan_model = None
        self.text2semantic_model = None
        self.decode_one_token_func = None

        self.reference_prompt_text_str = reference_prompt_text

        cprint("VoiceService Initialization Started...", 'blue')

        cprint("Initializing models...", 'cyan')
        self._initialize_models()

        ref_input_path = Path(reference_voice_input_path)
        cloned_features_dir = self._output_dir / "cloned_reference_voice"
        cloned_features_dir.mkdir(parents=True, exist_ok=True)
        cloned_features_basename = ref_input_path.stem + "_features"
        potential_cloned_features_path = cloned_features_dir / (cloned_features_basename + ".npy")

        if ref_input_path.suffix.lower() == ".npy":
            cprint(f"Using existing reference voice features: {ref_input_path}", 'cyan')
            if not ref_input_path.exists():
                raise FileNotFoundError(f"Provided reference .npy file not found: {ref_input_path}")
            self.cloned_voice_features_path = ref_input_path
        elif ref_input_path.suffix.lower() in [".wav", ".mp3", ".flac", ".ogg"]:
            cprint(f"Cloning reference voice from audio: {ref_input_path}", 'cyan')
            self.cloned_voice_features_path = potential_cloned_features_path
            self._perform_audio_to_features_cloning(
                input_audio_path=ref_input_path,
                output_features_npy_path=self.cloned_voice_features_path
            )
        else:
            raise ValueError(f"Unsupported reference voice file type: {ref_input_path}. Provide .npy or an audio file.")

        if not self.cloned_voice_features_path.exists():
            raise FileNotFoundError(f"Voice features not found or generated at {self.cloned_voice_features_path}")
        cprint(f"Using voice features from: {self.cloned_voice_features_path}", 'yellow')

        cprint("Warming up models...", 'cyan')
        self._warm_up_models()

        cprint("VoiceService initialized, voice processed, and models warmed up.", 'green')

    def _initialize_models(self):
        if self.vqgan_model is None:
            cprint("Loading VQGAN model...", 'magenta')
            load_start = time.time()
            self.vqgan_model = load_vqgan_model(
                config_name=self.config_name,
                checkpoint_path=str(self.vqgan_checkpoint_path),
                device=self.device
            )
            cprint(f"VQGAN model loaded in {time.time() - load_start:.2f} seconds.")
        
        if self.text2semantic_model is None:
            cprint("Loading Text2Semantic model...", 'magenta')
            load_start = time.time()
            self.text2semantic_model, self.decode_one_token_func = load_text2semantic_model(
                checkpoint_path=str(self.checkpoint_path),
                device=self.device,
                precision=self.precision,
                compile=True
            )
            cprint(f"Text2Semantic model loaded in {time.time() - load_start:.2f} seconds.")
        
            if self.device == "cuda":
                torch.cuda.synchronize()

            with torch.device(self.device):
                 self.text2semantic_model.setup_caches(
                    max_batch_size=1,
                    max_seq_len=self.text2semantic_model.config.max_seq_len,
                    dtype=next(self.text2semantic_model.parameters()).dtype,
                )

    def _perform_audio_to_features_cloning(self, input_audio_path: Path, output_features_npy_path: Path):
        cprint(f"Generating voice features from {input_audio_path} to {output_features_npy_path}", 'magenta')
        start_time = time.time()

        if self.vqgan_model is None:
            raise RuntimeError("VQGAN model not initialized before cloning.")

        output_audio_for_cloning_path = output_features_npy_path.with_suffix(".wav")

        vqgan_infer_main(
            input_path=Path(input_audio_path),
            output_path=Path(output_audio_for_cloning_path),
            checkpoint_path=str(self.vqgan_checkpoint_path),
            config_name=self.config_name,
            device=self.device,
            model=self.vqgan_model
        )
        
        if not output_features_npy_path.exists():
            # It's possible vqgan_infer_main saves with torch.save, and text2semantic expects np.load
            # This check ensures the file is created. We'll handle format conversion if needed later,
            # or assume vqgan_infer_main from the library should ideally save in np.load-compatible format if that's the ecosystem.
            # For now, we address the dummy file creation as the primary error source.
            # If cloning real audio also leads to this error, vqgan_infer_main's output format is the issue.
            raise FileNotFoundError(f"Cloned voice feature file {output_features_npy_path} was not created.")
        
        # Optional: Convert if vqgan_infer_main saved with torch.save and text2semantic needs np.load
        try:
            np.load(output_features_npy_path) # Test load
        except (TypeError, ValueError) as e: # Catch errors if it's not a valid NumPy .npy file
            cprint(f"Warning: Cloned feature file {output_features_npy_path} might not be in native NumPy format (Error: {e}). Attempting conversion.", "yellow")
            try:
                tensor_data = torch.load(output_features_npy_path, map_location='cpu')
                if isinstance(tensor_data, torch.Tensor):
                    np.save(output_features_npy_path, tensor_data.cpu().numpy()) # Overwrite with NumPy format
                    cprint(f"Successfully converted {output_features_npy_path} to NumPy format.", "green")
                else:
                    cprint(f"Error: Loaded data from {output_features_npy_path} is not a tensor, cannot convert.", "red")
            except Exception as conv_e:
                cprint(f"Error converting {output_features_npy_path} to NumPy format: {conv_e}", "red")
                raise # Re-raise if conversion fails, as it's critical for downstream use

        cprint(f"Voice features generated and saved to {output_features_npy_path} in {time.time() - start_time:.2f} seconds.", 'yellow')


    def _warm_up_models(self):
        cprint("Warming up models with a test inference...", 'magenta')
        warm_up_start = time.time()

        if self.text2semantic_model is None or self.vqgan_model is None or \
           not self.cloned_voice_features_path or not self.cloned_voice_features_path.exists():
            raise RuntimeError("Models not initialized or reference voice features not available for warm-up.")

        warm_up_text = "This is a brief warm-up sentence."
        warmup_output_dir = self._output_dir / "warmup_outputs"
        warmup_output_dir.mkdir(parents=True, exist_ok=True)

        cprint("Warming up Text2Semantic generation...", 'magenta')
        
        # Call generate_semantic_tokens and handle its current return type (tensor)
        returned_data_tensor = generate_semantic_tokens(
            text=warm_up_text,
            prompt_text=[self.reference_prompt_text_str],
            prompt_tokens=[str(self.cloned_voice_features_path)],
            checkpoint_path=str(self.checkpoint_path), # Passed but might be unused if model is provided
            half=(self.precision == torch.half),
            device=self.device,
            num_samples=1, 
            max_new_tokens=20,
            top_p=0.7, repetition_penalty=1.2, temperature=0.3,
            compile=False, # Model already compiled
            seed=42,
            iterative_prompt=True,
            chunk_length=30,
            output_dir=str(warmup_output_dir), # For text2semantic if it were to save
            model=self.text2semantic_model, # Passing the pre-loaded model
            decode_one_token=self.decode_one_token_func, # Passing the pre-loaded function
        )

        generated_semantic_token_paths = []
        if isinstance(returned_data_tensor, torch.Tensor):
            # Manually save the tensor as text2semantic.inference.main might not be saving it.
            # Assuming the tensor is for the first (and only, in warm-up) sample.
            semantic_tokens_save_path = warmup_output_dir / "codes_0.npy" # Standard naming
            np.save(semantic_tokens_save_path, returned_data_tensor.cpu().numpy())
            cprint(f"Warm-up: Manually saved semantic tokens to {semantic_tokens_save_path}", "magenta")
            generated_semantic_token_paths.append(semantic_tokens_save_path)
        elif returned_data_tensor is not None:
            cprint(f"Warm-up: generate_semantic_tokens returned unexpected data type: {type(returned_data_tensor)}", "red")
            cprint(f"Model warm-up failed at semantic token generation in {time.time() - warm_up_start:.2f} seconds", 'yellow')
            return
        else: # Is None
             cprint("Warm-up: Semantic token generation returned None. This might be due to text2semantic.inference.main's internal logic (e.g. specific 'idx' condition not met).", "red")
             cprint(f"Model warm-up failed at semantic token generation in {time.time() - warm_up_start:.2f} seconds", 'yellow')
             return


        if not generated_semantic_token_paths or not generated_semantic_token_paths[0].exists():
            cprint("Warm-up: Semantic token file not found after generation attempt.", "red")
            cprint(f"Model warm-up failed in {time.time() - warm_up_start:.2f} seconds", 'yellow')
            return

        semantic_tokens_for_vqgan_warmup = generated_semantic_token_paths[0]
        cprint(f"Warm-up: Using semantic tokens from: {semantic_tokens_for_vqgan_warmup}", 'magenta')

        cprint("Warming up VQGAN inference from codes...", 'magenta')
        vqgan_infer_main(
            input_path=Path(semantic_tokens_for_vqgan_warmup),
            output_path=Path(warmup_output_dir / "warmup_audio.wav"),
            checkpoint_path=Path(self.vqgan_checkpoint_path),
            config_name=self.config_name,
            device=self.device,
            model=self.vqgan_model
        )
        cprint(f"VQGAN warm-up pass completed. Warm-up audio saved in {warmup_output_dir}.", 'magenta')
        cprint(f"Model warm-up fully completed in {time.time() - warm_up_start:.2f} seconds.", 'yellow')


    def generate_speech(self, text_to_speak: str, output_audio_basename: str = "generated_speech", num_samples: int = 1):
        cprint(f"Starting speech generation for: \"{text_to_speak[:70]}...\"", 'blue')
        overall_start_time = time.time()

        if self.text2semantic_model is None or self.vqgan_model is None or \
           not self.cloned_voice_features_path or not self.cloned_voice_features_path.exists() or \
           not self.reference_prompt_text_str:
            raise RuntimeError("VoiceService not properly initialized or vital assets missing.")

        inference_codes_dir = self._output_dir / "tts_intermediate_codes" / output_audio_basename
        inference_codes_dir.mkdir(parents=True, exist_ok=True)
        
        t2s_start_time = time.time()
        cprint(f"Generating semantic tokens using reference prompt: \"{self.reference_prompt_text_str[:70]}...\" and features from: {self.cloned_voice_features_path.name}", 'magenta')
        
        # Adapt to current text2semantic.inference.main behavior
        returned_tensor_data = generate_semantic_tokens(
            text=text_to_speak,
            prompt_text=[self.reference_prompt_text_str],
            prompt_tokens=[str(self.cloned_voice_features_path)],
            checkpoint_path=str(self.checkpoint_path),
            half=(self.precision == torch.half),
            device=self.device,
            num_samples=num_samples, # Pass requested num_samples
            max_new_tokens=1024,
            top_p=0.7, repetition_penalty=1.2, temperature=0.3,
            compile=False,
            seed=42,
            iterative_prompt=True,
            chunk_length=100,
            output_dir=str(inference_codes_dir), # For text2semantic if it were to save
            model=self.text2semantic_model,
            decode_one_token=self.decode_one_token_func,
        )
        cprint(f"Semantic token generation (raw call) took {time.time() - t2s_start_time:.2f}s.", 'yellow')

        generated_semantic_paths = []
        if isinstance(returned_tensor_data, torch.Tensor):
            # text2semantic.inference.main currently returns one tensor due to "if idx == 1" logic.
            # We save this one tensor. This means we'll effectively process one sample.
            save_path = inference_codes_dir / "codes_0.npy" # Save as the first sample
            np.save(save_path, returned_tensor_data.cpu().numpy())
            generated_semantic_paths.append(save_path)
            cprint(f"Manually saved semantic tokens for one sample to {save_path}", "magenta")
            if num_samples > 1:
                cprint(f"Warning: Requested {num_samples} samples, but current 'generate_semantic_tokens' "
                       f"interface likely provided data for only one sample. Processing this one.", "yellow")
        elif returned_tensor_data is not None:
             cprint(f"generate_speech: generate_semantic_tokens returned unexpected data type: {type(returned_tensor_data)}", "red")
        else: # Is None
            cprint("generate_speech: Semantic token generation returned None. Cannot proceed with audio synthesis.", "red")


        if not generated_semantic_paths: # Check if any paths were successfully prepared
            cprint("Semantic token processing failed or produced no usable files.", "red")
            return []

        generated_audio_files = []
        total_audio_duration = 0
        vocoding_start_time = time.time()

        # Loop will run for 0 or 1 iteration based on current adaptation
        for i, semantic_token_path in enumerate(generated_semantic_paths):
            cprint(f"Vocoding sample {i+1}/{len(generated_semantic_paths)} from {semantic_token_path.name}...", 'magenta')
            sample_vocoding_start_time = time.time()
            if isinstance(semantic_token_path, (str, Path)):
                # Load numpy data from the path
                npy_tts_voice = np.load(semantic_token_path)
                npy_tts_voice = torch.from_numpy(npy_tts_voice).to(self.device).long()
            else:
                # Assume it's already a numpy array
                npy_tts_voice = semantic_token_path
            audio_data, audio_duration = vqgan_infer_tts_from_codes(
                npy_tts_voice=npy_tts_voice,
                device=self.device,
                model=self.vqgan_model
            )
            cprint(f"  - Vocoding for sample {i+1} took {time.time() - sample_vocoding_start_time:.2f}s.", 'magenta')
            
            # If multiple samples were genuinely produced by generate_semantic_tokens and saved:
            # suffix = f"_{i}" if num_samples > 1 (or len(generated_semantic_paths) > 1) else ""
            # For now, with one sample from text2semantic:
            suffix = "" 
            final_audio_filename = f"{output_audio_basename}{suffix}.wav"
            output_audio_path = self._output_dir / final_audio_filename
            
            sf.write(str(output_audio_path), audio_data[0, 0].float().detach().cpu().numpy(), 44100)
            
            cprint(f"  - Audio sample {i+1} ({audio_duration:.2f}s) saved to: {output_audio_path}", 'yellow')
            generated_audio_files.append(output_audio_path)
            total_audio_duration += audio_duration
        
        cprint(f"Total vocoding time for {len(generated_semantic_paths)} sample(s): {time.time() - vocoding_start_time:.2f}s.", 'yellow')
        
        overall_execution_time = time.time() - overall_start_time
        cprint(f"Speech generation process completed in {overall_execution_time:.2f} seconds.", 'blue')
        if total_audio_duration > 0 and overall_execution_time > 0:
            rtf = overall_execution_time / total_audio_duration
            cprint(f"Real-Time Factor (RTF): {rtf:.2f} (Speed: {1/rtf:.2f}x Real-Time)", 'green')
            
        return generated_audio_files


if __name__ == "__main__":
    INPUT_REFERENCE_VOICE = "./clone_voice_feature.npy"
    placeholder_npy_path = Path(INPUT_REFERENCE_VOICE)
    
    # Crucial Fix: Save dummy .npy file using np.save for compatibility with np.load
    if not placeholder_npy_path.exists():
        cprint(f"Warning: {placeholder_npy_path} not found. Creating a dummy .npy file for testing.", "red")
        # Create a NumPy array
        dummy_np_array = np.random.randint(0, 1024, (1, 200, 8)).astype(np.int16) # Example shape and type
        np.save(placeholder_npy_path, dummy_np_array) # Use np.save
        cprint(f"Dummy .npy file saved to {placeholder_npy_path} using np.save().", "green")

    REFERENCE_PROMPT = "This is the reference prompt text, guiding the style of the cloned voice."
    TEXT_TO_SYNTHESIZE = "Hello world, I am now speaking with the cloned voice characteristics. How do I sound?"

    try:
        cprint(f"Initializing VoiceService with reference: {INPUT_REFERENCE_VOICE}", 'blue')
        voice_service = VoiceService(
            reference_voice_input_path=INPUT_REFERENCE_VOICE,
            reference_prompt_text=REFERENCE_PROMPT
        )
        
        cprint(f"\nGenerating speech for: \"{TEXT_TO_SYNTHESIZE}\"", 'blue')
        generated_files = voice_service.generate_speech(
            text_to_speak=TEXT_TO_SYNTHESIZE,
            output_audio_basename="my_cloned_speech",
            num_samples=1 
        )

        if generated_files:
            cprint(f"\nSuccessfully generated audio files:", 'green')
            for f_path in generated_files:
                cprint(f" - {f_path}", 'green')
        else:
            cprint("\nSpeech generation failed or produced no audio files.", 'red')

    except FileNotFoundError as e:
        cprint(f"ERROR: A required file was not found: {e}", "red")
        cprint("Please ensure all paths (checkpoints, reference voice) are correct.", "red")
    except RuntimeError as e:
        cprint(f"ERROR: A runtime error occurred: {e}", "red")
    except Exception as e:
        cprint(f"An unexpected error occurred: {e}", "red")
        import traceback
        traceback.print_exc()