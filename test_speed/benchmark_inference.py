from pathlib import Path
import shutil
import zipfile

import numpy as np
import soundfile as sf
import torch
from tqdm import tqdm

from tinfer.core.request import AlignmentType
from tinfer.models.base.model import IntermediateRepresentation
from tinfer.models.impl.styletts2.model.inference_config import StyleTTS2Params
from tinfer.models.impl.styletts2.model.model import StyleTTS2

from test_speed.benchmark_data import (
    PhonemeMetric,
    ReferenceDuration,
    RequestMetric,
    TextInput,
)


def archive_wav_names(archive_path: Path) -> list[str]:
    with zipfile.ZipFile(archive_path) as archive:
        names = [
            Path(info.filename).name
            for info in archive.infolist()
            if not info.is_dir()
            and info.filename.lower().endswith(".wav")
            and "__MACOSX" not in Path(info.filename).parts
        ]
    if len(names) != len(set(names)):
        raise ValueError("Archive contains duplicate WAV basenames")
    return sorted(names)


def extract_selected(
    archive_path: Path,
    selected_names: list[str],
    output_dir: Path,
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    selected = set(selected_names)
    with zipfile.ZipFile(archive_path) as archive:
        infos = {
            Path(info.filename).name: info
            for info in archive.infolist()
            if Path(info.filename).name in selected
            and "__MACOSX" not in Path(info.filename).parts
        }
        if set(infos) != selected:
            missing = sorted(selected - set(infos))
            raise FileNotFoundError(f"Selected archive WAVs are missing: {missing}")
        paths = []
        for name in tqdm(selected_names, desc="Extract references", unit="wav"):
            destination = output_dir / name
            with archive.open(infos[name]) as source, destination.open("wb") as target:
                shutil.copyfileobj(source, target)
            paths.append(destination)
    return paths


def measure_reference_durations(
    reference_paths: list[Path],
) -> list[ReferenceDuration]:
    return [
        ReferenceDuration(path.stem, sf.info(path).duration)
        for path in reference_paths
    ]


def measure_result(
    result: IntermediateRepresentation,
    voice_id: str,
    text_input: TextInput,
    audio_path: Path,
) -> tuple[RequestMetric, list[PhonemeMetric]]:
    if "window_count" in result.metadata:
        raise RuntimeError(f"Model internally chunked {text_input.text_id}")
    alignments = result.metadata["word_alignments"]
    if not alignments:
        raise RuntimeError(f"No predictor alignments for {text_input.text_id}")

    phonemes = [
        PhonemeMetric(
            phoneme=item.item,
            duration_seconds=(item.end_ms - item.start_ms) / 1000.0,
            voice_id=voice_id,
            text_id=text_input.text_id,
        )
        for item in alignments
        if item.item and not item.item.isspace()
    ]
    predicted_seconds = sum(item.duration_seconds for item in phonemes)
    if predicted_seconds <= 0:
        raise RuntimeError(f"Non-positive predictor duration for {text_input.text_id}")

    request = RequestMetric(
        voice_id=voice_id,
        text_id=text_input.text_id,
        text=text_input.text,
        text_length=len(text_input.text),
        input_phoneme_tokens=text_input.input_phoneme_tokens,
        phoneme_count=len(phonemes),
        predicted_seconds=predicted_seconds,
        phonemes_per_second=len(phonemes) / predicted_seconds,
        audio_path=str(audio_path),
    )
    return request, phonemes


def load_model(model_path: Path, device: str) -> StyleTTS2:
    if not model_path.is_file():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    model = StyleTTS2(device=device)
    model.load(
        str(model_path),
        device=device,
        compile_model=False,
        load_style_encoder=True,
    )
    return model


def embed_references(
    model: StyleTTS2,
    reference_paths: list[Path],
    output_dir: Path,
) -> list[str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    voice_ids = []
    for reference_path in tqdm(reference_paths, desc="Embed voices", unit="voice"):
        voice_id = reference_path.stem
        model.load_voice_from_audio(voice_id, str(reference_path))
        voice_vector = model.get_voice(voice_id)
        torch.save({"voice_vector": voice_vector.detach().cpu()}, output_dir / f"{voice_id}.pth")
        voice_ids.append(voice_id)
    return voice_ids


def synthesize_all(
    model: StyleTTS2,
    voice_ids: list[str],
    text_inputs: list[TextInput],
    output_dir: Path,
    use_diffusion: bool,
) -> tuple[list[RequestMetric], list[PhonemeMetric]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    request_metrics = []
    phoneme_metrics = []
    pairs = [(voice_id, text_input) for voice_id in voice_ids for text_input in text_inputs]
    for voice_id, text_input in tqdm(pairs, desc="Synthesize", unit="audio"):
        style_params = StyleTTS2Params(use_diffusion=use_diffusion)
        if model._text_token_count(text_input.text, style_params) > model._max_styletts_tokens:
            raise RuntimeError(f"Input exceeds one model window: {text_input.text_id}")
        result = model.generate(
            text_input.text,
            {"voice_id": voice_id},
            {"use_diffusion": use_diffusion},
            {"alignment_type": AlignmentType.PHONEME},
        )
        voice_dir = output_dir / voice_id
        voice_dir.mkdir(parents=True, exist_ok=True)
        audio_path = voice_dir / f"{text_input.text_id}.wav"
        sf.write(audio_path, np.asarray(result.data, dtype=np.float32), result.sample_rate)
        request, phonemes = measure_result(result, voice_id, text_input, audio_path)
        request_metrics.append(request)
        phoneme_metrics.extend(phonemes)
    return request_metrics, phoneme_metrics
