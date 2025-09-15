from models.fish_speech.models.text2semantic.llama import (
    BaseTransformer,
    DualARTransformer,
    NaiveTransformer,
)
from typing import Literal, Optional, Tuple, Union
from loguru import logger
import torch

from models.fish_speech.models.text2semantic.inference import (
    decode_one_token_ar,
    decode_one_token_ar_agent,
    decode_one_token_naive,
    decode_one_token_naive_agent,
)

def load_model(checkpoint_path, device, precision, compile=False, is_agent=False):
    model: Union[NaiveTransformer, DualARTransformer] = BaseTransformer.from_pretrained(
        checkpoint_path, load_weights=True, is_agent=is_agent
    )

    model = model.to(device=device, dtype=precision)
    logger.info(f"Restored model from checkpoint")

    if isinstance(model, DualARTransformer):
        decode_one_token = (
            decode_one_token_ar_agent if is_agent else decode_one_token_ar
        )
        logger.info("Using DualARTransformer")
    else:
        decode_one_token = (
            decode_one_token_naive_agent if is_agent else decode_one_token_naive
        )
        logger.info("Using NaiveTransformer")

    if compile:
        logger.info("Compiling function...")
        decode_one_token = torch.compile(
            decode_one_token,
            fullgraph=True,
            backend="inductor" if torch.cuda.is_available() else "aot_eager",
            mode="reduce-overhead" if torch.cuda.is_available() else None,
        )

    return model.eval(), decode_one_token

