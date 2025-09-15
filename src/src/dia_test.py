
"""Dia is a 1.6B parameter text to speech model"""
from dia.model import Dia
import time

model = Dia.from_pretrained("nari-labs/Dia-1.6B", compute_dtype="float16")

text = "[S1] Dia is an open weights text to dialogue model. [S2] You get full control over scripts and voices. [S1] Wow. Amazing. (laughs) [S2] Try it now on Git hub or Hugging Face."

ts = time.time()
output = model.generate(text, use_torch_compile=True, verbose=True)
print(f"Generation took {time.time() - ts:.2f} seconds")
model.save_audio("simple.mp3", output)