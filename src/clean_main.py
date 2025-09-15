"""Fish TTS example for generating speech using a cloned voice."""
import os
import time
import sys
from pathlib import Path
from typing import List, Optional, Union
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
    def __init__(self, 
                 checkpoint_dir: str = "checkpoints",
                 output_dir: str = "outputs",
                 cloned_voice_features_path: Union[str, Path] = "cloned_voice_features.npy",
                 device: Optional[str] = None,
                 precision: str = "half"):
        self._output_dir = Path(output_dir) 
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self.cloned_voice_features_path = Path(cloned_voice_features_path)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.precision = torch.half if precision == "half" else torch.float32
        
        self.checkpoint_dir = Path(checkpoint_dir)
        self.vqgan_checkpoint = self.checkpoint_dir / "firefly-gan-vq-fsq-8x1024-21hz-generator.pth"
        self.config_name = "firefly_gan_vq"

        self.vqgan_model = None
        self.text2semantic_model = None
        self.decode_one_token_func = None

        self.reference_prompt_text_str = "This is the reference prompt text, guiding the style of the cloned voice."

        self._initialize_models()

        print("Warming up models...")
        self._warm_up_models()

        print("VoiceService initialized, voice processed, and models warmed up.")

    def _initialize_models(self):
        if self.vqgan_model is None:
            load_start = time.time()
            self.vqgan_model = load_vqgan_model(
                config_name=self.config_name,
                checkpoint_path=str(self.vqgan_checkpoint),
                device=self.device
            )
            print(f"VQGAN model loaded in {time.time() - load_start:.2f} seconds.")
        
        if self.text2semantic_model is None:
            load_start = time.time()
            self.text2semantic_model, self.decode_one_token_func = load_text2semantic_model(
                checkpoint_path=str(self.checkpoint_dir),
                device=self.device,
                precision=self.precision,
                compile=True
            )
            print(f"Text2Semantic model loaded in {time.time() - load_start:.2f} seconds.")
        
            if self.device == "cuda":
                torch.cuda.synchronize()

            with torch.device(self.device):
                 self.text2semantic_model.setup_caches(
                    max_batch_size=1,
                    max_seq_len=self.text2semantic_model.config.max_seq_len,
                    dtype=next(self.text2semantic_model.parameters()).dtype,
                )

    def _perform_audio_to_features_cloning(self, input_audio_path: Path, output_features_npy_path: Path):
        print(f"Generating voice features from {input_audio_path} to {output_features_npy_path}")
        start_time = time.time()

        if self.vqgan_model is None:
            raise RuntimeError("VQGAN model not initialized before cloning.")

        output_audio_for_cloning_path = output_features_npy_path.with_suffix(".wav")

        vqgan_infer_main(
            input_path=Path(input_audio_path),
            output_path=Path(output_audio_for_cloning_path),
            checkpoint_path=str(self.vqgan_checkpoint),
            config_name=self.config_name,
            device=self.device,
            model=self.vqgan_model
        )
        
        if not output_features_npy_path.exists():
            raise FileNotFoundError(f"Cloned voice feature file {output_features_npy_path} was not created.")
        
        # Optional: Convert if vqgan_infer_main saved with torch.save and text2semantic needs np.load
        try:
            np.load(output_features_npy_path) # Test load
        except (TypeError, ValueError) as e: # Catch errors if it's not a valid NumPy .npy file
            print(f"Warning: Cloned feature file {output_features_npy_path} might not be in native NumPy format (Error: {e}). Attempting conversion.")
            try:
                tensor_data = torch.load(output_features_npy_path, map_location='cpu')
                if isinstance(tensor_data, torch.Tensor):
                    np.save(output_features_npy_path, tensor_data.cpu().numpy()) # Overwrite with NumPy format
                    print(f"Successfully converted {output_features_npy_path} to NumPy format.")
                else:
                    print(f"Error: Loaded data from {output_features_npy_path} is not a tensor, cannot convert.")
            except Exception as conv_e:
                print(f"Error converting {output_features_npy_path} to NumPy format: {conv_e}")
                raise # Re-raise if conversion fails, as it's critical for downstream use

        print(f"Voice features generated and saved to {output_features_npy_path} in {time.time() - start_time:.2f} seconds.")


    def _warm_up_models(self):

        if self.text2semantic_model is None or self.vqgan_model is None or \
           not self.cloned_voice_features_path or not self.cloned_voice_features_path.exists():
            raise RuntimeError("Models not initialized or reference voice features not available for warm-up.")

        warm_up_text = "This is a brief warm-up sentence."
        warmup_output_dir = self._output_dir / "warmup_outputs"
        warmup_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Call generate_semantic_tokens and handle its current return type (tensor)
        returned_data_tensor = generate_semantic_tokens(
            text=warm_up_text,
            prompt_text=[self.reference_prompt_text_str],
            prompt_tokens=[str(self.cloned_voice_features_path)],
            checkpoint_path=str(self.checkpoint_dir), # Passed but might be unused if model is provided
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
            generated_semantic_token_paths.append(semantic_tokens_save_path)
        elif returned_data_tensor is not None:
            return
        else: # Is None
             return


        if not generated_semantic_token_paths or not generated_semantic_token_paths[0].exists():
            return

        semantic_tokens_for_vqgan_warmup = generated_semantic_token_paths[0]

        vqgan_infer_main(
            input_path=Path(semantic_tokens_for_vqgan_warmup),
            output_path=Path(warmup_output_dir / "warmup_audio.wav"),
            checkpoint_path=Path(self.vqgan_checkpoint),
            config_name=self.config_name,
            device=self.device,
            model=self.vqgan_model
        )


    def generate_speech(self, text_to_speak: str, output_audio_basename: str = "generated_speech", num_samples: int = 1):
        overall_start_time = time.time()

        if self.text2semantic_model is None or self.vqgan_model is None or \
           not self.cloned_voice_features_path or not self.cloned_voice_features_path.exists() or \
           not self.reference_prompt_text_str:
            raise RuntimeError("VoiceService not properly initialized or vital assets missing.")

        inference_codes_dir = self._output_dir / "tts_intermediate_codes" / output_audio_basename
        inference_codes_dir.mkdir(parents=True, exist_ok=True)
        
        # Adapt to current text2semantic.inference.main behavior
        returned_tensor_data = generate_semantic_tokens(
            text=text_to_speak,
            prompt_text=[self.reference_prompt_text_str],
            prompt_tokens=[str(self.cloned_voice_features_path)],
            checkpoint_path=str(self.checkpoint_dir),
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

        generated_semantic_paths = []
        if isinstance(returned_tensor_data, torch.Tensor):
            # text2semantic.inference.main currently returns one tensor due to "if idx == 1" logic.
            # We save this one tensor. This means we'll effectively process one sample.
            save_path = inference_codes_dir / "codes_0.npy" # Save as the first sample
            np.save(save_path, returned_tensor_data.cpu().numpy())
            generated_semantic_paths.append(save_path)
            if num_samples > 1:
                print(f"Warning: Requested {num_samples} samples, but current 'generate_semantic_tokens' "
                       f"interface likely provided data for only one sample. Processing this one.")
        elif returned_tensor_data is not None:
             print(f"generate_speech: generate_semantic_tokens returned unexpected data type: {type(returned_tensor_data)}")
        else: # Is None
            print("generate_speech: Semantic token generation returned None. Cannot proceed with audio synthesis.")


        if not generated_semantic_paths: # Check if any paths were successfully prepared
            print("Semantic token processing failed or produced no usable files.")
            return []

        generated_audio_files = []
        total_audio_duration = 0
        vocoding_start_time = time.time()

        # Loop will run for 0 or 1 iteration based on current adaptation
        for i, semantic_token_path in enumerate(generated_semantic_paths):
            print(f"Vocoding sample {i+1}/{len(generated_semantic_paths)} from {semantic_token_path.name}...")
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
            print(f"  - Vocoding for sample {i+1} took {time.time() - sample_vocoding_start_time:.2f}s.")
            
            # If multiple samples were genuinely produced by generate_semantic_tokens and saved:
            # suffix = f"_{i}" if num_samples > 1 (or len(generated_semantic_paths) > 1) else ""
            # For now, with one sample from text2semantic:
            suffix = "" 
            final_audio_filename = f"{output_audio_basename}{suffix}.wav"
            output_audio_path = self._output_dir / final_audio_filename
            
            sf.write(str(output_audio_path), audio_data[0, 0].float().detach().cpu().numpy(), 44100)
            
            print(f"  - Audio sample {i+1} ({audio_duration:.2f}s) saved to: {output_audio_path}")
            generated_audio_files.append(output_audio_path)
            total_audio_duration += audio_duration
        
        print(f"Total vocoding time for {len(generated_semantic_paths)} sample(s): {time.time() - vocoding_start_time:.2f}s.")
        
        overall_execution_time = time.time() - overall_start_time
        print(f"Speech generation process completed in {overall_execution_time:.2f} seconds.")
        if total_audio_duration > 0 and overall_execution_time > 0:
            rtf = overall_execution_time / total_audio_duration
            print(f"Real-Time Factor (RTF): {rtf:.2f} (Speed: {1/rtf:.2f}x Real-Time)")
            
        return generated_audio_files


if __name__ == "__main__":

    TEXT_TO_SYNTHESIZE = "Hello world, I am now speaking with the cloned voice characteristics. How do I sound?"
    INPUT_REFERENCE_VOICE = "/home/ubuntu/ai_service/clone_voice_feature.npy"
    try:
        print(f"Initializing VoiceService with reference: {INPUT_REFERENCE_VOICE}")
        voice_service = VoiceService(
            checkpoint_dir="/home/ubuntu/ai_service/checkpoints",
            output_dir="/home/ubuntu/ai_service/outputs",
            cloned_voice_features_path=INPUT_REFERENCE_VOICE,
            device="cuda" if torch.cuda.is_available() else "cpu",
            precision="half" 
        )
        
        print(f"\nGenerating speech for: \"{TEXT_TO_SYNTHESIZE}\"")
        generated_files = voice_service.generate_speech(
            text_to_speak=TEXT_TO_SYNTHESIZE,
            output_audio_basename="my_cloned_speech",
            num_samples=1 
        )

        if generated_files:
            print(f"\nSuccessfully generated audio files:")
            for f_path in generated_files:
                print(f" - {f_path}")
        else:
            print("\nSpeech generation failed or produced no audio files.")

    except FileNotFoundError as e:
        print(f"ERROR: A required file was not found: {e}")
        print("Please ensure all paths (checkpoints, reference voice) are correct.", "red")
    except RuntimeError as e:
        print(f"ERROR: A runtime error occurred: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()