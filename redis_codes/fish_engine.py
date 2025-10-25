from pathlib import Path
import os
import click
import hydra
import numpy as np
import pyrootutils
import soundfile as sf
import torch
import torchaudio
from hydra import compose, initialize
from hydra.utils import instantiate
from loguru import logger
from text2semantic import inference_text2semantic
from text2semantic import init_model as init_encoder
from fish_speech.models.dac.inference import main as vqgan_infer_main

# pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
AUDIO_EXTENSIONS = {
    ".mp3",
    ".wav",
    ".flac",
    ".ogg",
    ".m4a",
    ".wma",
    ".aac",
    ".aiff",
    ".aif",
    ".aifc",
}

class FishTextToSpeech:
    def __init__(self, config_name:str, checkpoint_path:str, device:str="cuda"):
        self.config_name = config_name
        self.checkpoint_path = checkpoint_path
        self.device = device
        self.decoder = None
        self.encoder = None
        self.decode_one_token = None
        
    def load_model(self):
        hydra.core.global_hydra.GlobalHydra.instance().clear()
        with initialize(version_base="1.3", config_path="."):
            cfg = compose(config_name=self.config_name)

        self.encoder = instantiate(cfg)
        state_dict = torch.load(self.checkpoint_path, map_location=self.device, mmap=True, weights_only=True)
        if "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]

        if any("generator" in k for k in state_dict):
            state_dict = {
                k.replace("generator.", ""): v
                for k, v in state_dict.items()
                if "generator." in k
            }

        result = self.encoder.load_state_dict(state_dict, strict=False, assign=True)
        self.encoder.eval()
        self.encoder.to(self.device)
        logger.info(f"Loaded encoder: {result}")
        
        
        self.decoder, self.decode_one_token = init_encoder(Path(self.checkpoint_path).parent, self.device, torch.bfloat16, compile=False)
        with torch.device(self.device):
            self.decoder.setup_caches(
                max_batch_size=1,
                max_seq_len=self.decoder.config.max_seq_len,
                dtype=next(self.decoder.parameters()).dtype,
            )
        logger.info("Loading decoder ...")
        # self._warm_up_models()
        
    
    def _warm_up_models(self):
        if self.decoder is None or self.encoder is None:
            raise RuntimeError("Models not initialized or reference voice features not available for warm-up.")

        warm_up_text = "یہ ایک مختصر وارم اپ جملہ ہے۔"
        warmup_output_dir = "./test"
        os.makedirs(warmup_output_dir, exist_ok=True)
        # Call generate_semantic_tokens and handle its current return type (tensor)
        prompt_text = open("./test/ref/reference_text.txt", "r").readlines()[0]
        returned_data_tensor = inference_text2semantic(
        decoder_model=self.decoder,
        decode_one_token=self.decode_one_token,
        text=warm_up_text,
        prompt_text=(prompt_text, ),
        prompt_tokens=("./test/ref/reference_codec.npy", ),
        num_samples=1,
        max_new_tokens=1024,
        top_p=0.9,
        repetition_penalty=1.0,
        temperature=0.8,
        device="cuda",
        compile=False,
        seed=42,
        iterative_prompt=True,
        chunk_length=300,
        output_dir="./test")

        generated_semantic_token_paths = []
        if isinstance(returned_data_tensor, torch.Tensor):
            # Manually save the tensor as text2semantic.inference.main might not be saving it.
            # Assuming the tensor is for the first (and only, in warm-up) sample.
            semantic_tokens_save_path = warmup_output_dir / "codes_0.npy" # Standard naming
            np.save(semantic_tokens_save_path, returned_data_tensor.cpu().numpy())
            generated_semantic_token_paths.append(semantic_tokens_save_path)
        elif returned_data_tensor is not None:
            return
        else:
             return


        if not generated_semantic_token_paths or not generated_semantic_token_paths[0].exists():
            return

        semantic_tokens_for_vqgan_warmup = generated_semantic_token_paths[0]

        vqgan_infer_main(
            input_path=Path(semantic_tokens_for_vqgan_warmup),
            output_path=Path(warmup_output_dir / "warmup_audio.wav"),
            checkpoint_path=Path(self.vqgan_checkpoint_path),
            config_name=self.config_name,
            device=self.device,
            model=self.encoder
        )
        print(f"VQGAN warm-up pass completed. Warm-up audio saved in {warmup_output_dir}.", 'magenta')

    @torch.no_grad()
    def voice_cloning(self, input_path:str, input_text: str, output_dir_path:str="./test", save_output:bool=True):
        """
        input_path str: path of reference voice
        output_path str: output directory path of codec file and text file
        input_text str: context of input voice
        """
        assert os.path.isfile(input_path), f"input_path parameter should be a directory but {input_path} is file"
        # assert os.path.isfile(output_dir_path), f"output_path parameter should be a directory but {output_dir_path} is file"
        os.makedirs(output_dir_path, exist_ok=True)
        text_file_path = os.path.join(output_dir_path, "reference_text.txt")
        voice_file_path = os.path.join(output_dir_path, "reference_codec.npy")
        
        # write input text
        f = open(text_file_path, "w")
        f.write(input_text)
        
        logger.info(f"Processing voice cloning on: {input_path}")
        # Load audio
        audio, sr = torchaudio.load(str(input_path))
        if audio.shape[0] > 1:
            audio = audio.mean(0, keepdim=True)
        audio = torchaudio.functional.resample(audio, sr, self.encoder.sample_rate)

        audios = audio[None].to(self.device)
        logger.info(f"Loaded audio with {audios.shape[2] / self.encoder.sample_rate:.2f} seconds")

        # VQ Encoder
        audio_lengths = torch.tensor([audios.shape[2]], device=self.device, dtype=torch.long)
        indices, indices_lens = self.encoder.encode(audios, audio_lengths)

        if indices.ndim == 3:
            indices = indices[0]

        logger.info(f"Generated indices of shape {indices.shape}")

        # Save indices
        if save_output:
            np.save(voice_file_path, indices.cpu().numpy())
        return (f, indices.cpu().numpy())

    @torch.no_grad()
    def synthesize_voice(self, text:str, output_codec_path_dir:str, reference_dir:str="./test", output_wav_path:str="./test", save_output:bool=True):
        """
        output_codec_path_dir str: codec numpy path
        output_wav_path str: output wav path
        output_codec_path str: output codec path
        text str: input user text
        prompt_text str:
        prompt_tokens str:
        """
        assert os.path.isdir(reference_dir), "can not find reference directory"
        # assert os.path.isdir(output_codec_path_dir), "output_codec_path_dir should be a directory not a file."
        os.makedirs(output_codec_path_dir, exist_ok=True)
        
        reference_text_path = os.path.join(reference_dir, "reference_text.txt")
        reference_codec_path = os.path.join(reference_dir, "reference_codec.npy")
        prompt_text = open(reference_text_path, "r").readlines()[0]
        inference_text2semantic(
        decoder_model=self.decoder,
        decode_one_token=self.decode_one_token,
        text=text,
        prompt_text=(prompt_text, ),
        prompt_tokens=(reference_codec_path, ),
        num_samples=1,
        max_new_tokens=1024,
        top_p=0.9,
        repetition_penalty=1.0,
        temperature=0.8,
        device="cuda",
        compile=False,
        seed=42,
        iterative_prompt=True,
        chunk_length=300,
        output_dir=output_codec_path_dir)
        
        input_path = os.path.join(output_codec_path_dir, "codes_0.npy")
        logger.info(f"Processing precomputed indices from {input_path}")
        indices = np.load(input_path)
        indices = torch.from_numpy(indices).to(self.device).long()
        assert indices.ndim == 2, f"Expected 2D indices, got {indices.ndim}"
        indices_lens = torch.tensor([indices.shape[1]], device=self.device, dtype=torch.long)
        # Restore
        fake_audios, audio_lengths = self.encoder.decode(indices, indices_lens)
        audio_time = fake_audios.shape[-1] / self.encoder.sample_rate
        logger.info(f"Generated audio of shape {fake_audios.shape}, equivalent to {audio_time:.2f} seconds from {indices.shape[1]} features, features/second: {indices.shape[1] / audio_time:.2f}")
        # Save audio
        fake_audio = fake_audios[0, 0].float().cpu().numpy()
        if save_output:
            sf.write(output_wav_path, fake_audio, self.encoder.sample_rate)
            logger.info(f"Saved audio to {output_wav_path}")
        return (fake_audio, self.encoder.sample_rate)
        
    



def load_model(config_name, checkpoint_path, device):
    hydra.core.global_hydra.GlobalHydra.instance().clear()
    with initialize(version_base="1.3", config_path="."):
        cfg = compose(config_name=config_name)

    model = instantiate(cfg)
    state_dict = torch.load(checkpoint_path, map_location=device, mmap=True, weights_only=True)
    if "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]

    if any("generator" in k for k in state_dict):
        state_dict = {
            k.replace("generator.", ""): v
            for k, v in state_dict.items()
            if "generator." in k
        }

    result = model.load_state_dict(state_dict, strict=False, assign=True)
    model.eval()
    model.to(device)

    logger.info(f"Loaded model: {result}")
    return model

@torch.no_grad()
def inference_dac(input_path, output_path, config_name, checkpoint_path, device):
    model = load_model(config_name, checkpoint_path, device=device)

    if input_path.suffix in AUDIO_EXTENSIONS:
        print("1"*10)
        logger.info(f"Processing in-place reconstruction of {input_path}")

        # Load audio
        audio, sr = torchaudio.load(str(input_path))
        if audio.shape[0] > 1:
            audio = audio.mean(0, keepdim=True)
        audio = torchaudio.functional.resample(audio, sr, model.sample_rate)

        audios = audio[None].to(device)
        logger.info(
            f"Loaded audio with {audios.shape[2] / model.sample_rate:.2f} seconds"
        )

        # VQ Encoder
        audio_lengths = torch.tensor([audios.shape[2]], device=device, dtype=torch.long)
        indices, indices_lens = model.encode(audios, audio_lengths)

        if indices.ndim == 3:
            indices = indices[0]

        logger.info(f"Generated indices of shape {indices.shape}")

        # Save indices
        np.save(output_path.with_suffix(".npy"), indices.cpu().numpy())
    elif input_path.suffix == ".npy":
        print("2"*10)
        logger.info(f"Processing precomputed indices from {input_path}")
        indices = np.load(input_path)
        indices = torch.from_numpy(indices).to(device).long()
        assert indices.ndim == 2, f"Expected 2D indices, got {indices.ndim}"
        indices_lens = torch.tensor([indices.shape[1]], device=device, dtype=torch.long)
    else:
        raise ValueError(f"Unknown input type: {input_path}")

    # Restore
    fake_audios, audio_lengths = model.decode(indices, indices_lens)
    audio_time = fake_audios.shape[-1] / model.sample_rate

    logger.info(
        f"Generated audio of shape {fake_audios.shape}, equivalent to {audio_time:.2f} seconds from {indices.shape[1]} features, features/second: {indices.shape[1] / audio_time:.2f}"
    )

    # Save audio
    fake_audio = fake_audios[0, 0].float().cpu().numpy()
    sf.write(output_path, fake_audio, model.sample_rate)
    logger.info(f"Saved audio to {output_path}")


if __name__ == "__main__":
    input_path = Path("voice_cloning/asil_omran_38s.wav")
    output_path = Path("fake.wav")
    config_name = "modded_dac_vq"
    checkpoint_path = "checkpoints/openaudio-s1-mini/codec.pth"
    device = "cuda"
    inference_dac(input_path, output_path, config_name, checkpoint_path, device)
    
