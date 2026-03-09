from pathlib import Path
import os

from tinfer.core.engine import StreamingTTS
from tinfer.config.engine_config import StreamingTTSConfig
from config import model_id, model_path, voices_folder, voice_id

def load_model(config=None, warmup_kwargs=None):
    if config is None:
        config = StreamingTTSConfig(compile_models=False)

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    if not os.path.exists(voices_folder):
        raise FileNotFoundError(f"Voices folder not found: {voices_folder}")

    tts = StreamingTTS(config)
    print(f"Loading model from {model_path}")
    tts.load_model(model_id, str(model_path), voices_folder=voices_folder)
    tts.warmup([model_id], [voice_id], **(warmup_kwargs or {}))
    return tts
