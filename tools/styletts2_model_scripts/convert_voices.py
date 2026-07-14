from __future__ import annotations

import argparse
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

import numpy as np
import soundfile as sf
import torch
import yaml

REPOSITORY = Path(__file__).resolve().parents[2]
if str(REPOSITORY) not in sys.path:
    sys.path.insert(0, str(REPOSITORY))

from tinfer.models.impl.styletts2.model.modules.load_utils import load_original_styletts2_model
from tinfer.models.impl.styletts2.voice.encoder import StyleTTS2VoiceEncoder
from tools.styletts2_model_scripts.artifacts import write_tinf
from tools.styletts2_model_scripts.convert_model import find_model_files, resolve_config_paths


def mono_waveform(waveform: np.ndarray) -> np.ndarray:
    if waveform.ndim == 2:
        waveform = waveform.mean(axis=1)
    if waveform.ndim != 1 or waveform.size == 0:
        raise ValueError("WAV must contain non-empty mono or interleaved-channel audio")
    waveform = np.asarray(waveform, dtype=np.float32)
    if not np.isfinite(waveform).all():
        raise ValueError("WAV samples must be finite")
    return waveform


def validate_voice_vector(vector: torch.Tensor) -> np.ndarray:
    value = vector.detach().float().cpu().numpy()
    if value.size != 256:
        raise ValueError(f"StyleTTS2 voice vector must contain 256 values, got {value.size}")
    value = np.ascontiguousarray(value.reshape(256), dtype=np.float32)
    if not np.isfinite(value).all():
        raise ValueError("StyleTTS2 voice vector must be finite")
    return value


def convert_voices(model_folder: Path, wav_files: tuple[Path, ...], output: Path) -> None:
    names = [wav.stem for wav in wav_files]
    if not wav_files:
        raise ValueError("at least one WAV file is required")
    if len(names) != len(set(names)):
        raise ValueError("duplicate voice name from WAV file stems")
    for wav in wav_files:
        if not wav.is_file() or wav.suffix.lower() != ".wav":
            raise FileNotFoundError(f"WAV file does not exist: {wav}")

    checkpoint, config_path = find_model_files(model_folder)
    source_config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(source_config, dict):
        raise ValueError("StyleTTS2 YAML config must contain a mapping")
    resolved = resolve_config_paths(source_config, model_folder)
    with TemporaryDirectory(prefix="tinfer-styletts2-voice-") as temporary:
        resolved_path = Path(temporary) / "config.yml"
        resolved_path.write_text(yaml.safe_dump(resolved, sort_keys=False), encoding="utf-8")
        model, model_config = load_original_styletts2_model(str(checkpoint), str(resolved_path))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    for component in model.values():
        component.to(device).eval()
    encoder = StyleTTS2VoiceEncoder(model, device, model_config.preprocess.sr)
    output.mkdir(parents=True, exist_ok=True)
    for wav in wav_files:
        waveform, sample_rate = sf.read(wav, dtype="float32")
        vector = validate_voice_vector(encoder.compute_style_from_waveform(mono_waveform(waveform), sample_rate))
        write_tinf(output / f"{wav.stem}.tinf", {"ref_s": vector})


def main() -> None:
    parser = argparse.ArgumentParser(description="Export WAV voice embeddings for a StyleTTS2 model")
    parser.add_argument("model_folder", type=Path)
    parser.add_argument("wav", type=Path, nargs="+")
    parser.add_argument("-o", "--output", type=Path, required=True)
    arguments = parser.parse_args()
    convert_voices(arguments.model_folder, tuple(arguments.wav), arguments.output)


if __name__ == "__main__":
    main()
