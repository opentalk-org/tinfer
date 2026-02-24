from pathlib import Path
import os

from tinfer.core.engine import StreamingTTS
from tinfer.config.engine_config import StreamingTTSConfig

base_dir = Path(__file__).parent.parent.parent.parent
model_id = "styletts2"
model_name = "magda"
voice_id = "magda_001"
model_path = base_dir / "converted_models" / model_name / "model.pth"
voices_folder = str(base_dir / "converted_models" / model_name / "voices")


def load_model(config=None, warmup_kwargs=None):
    if config is None:
        config = StreamingTTSConfig()

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    if not os.path.exists(voices_folder):
        raise FileNotFoundError(f"Voices folder not found: {voices_folder}")

    tts = StreamingTTS(config)
    print(f"Loading model from {model_path}")
    tts.load_model(model_id, str(model_path), voices_folder=voices_folder)
    tts.warmup([model_id], [voice_id], **(warmup_kwargs or {}))
    return tts
