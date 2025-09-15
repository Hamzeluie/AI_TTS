import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from models.fish_speech.voice import VoiceService



vs = VoiceService()

# vs.clone_voice()
vs.fishspeech(
    text = "I'm borhan. I'm curious robot.how can I help you today? i think you need help!",
    clone_npy_path="/root/m15kh/ai_service/outputs/clone_voice_feature.npy"  # Use this `.npy` file to apply the cloned voice to the given text.
)
