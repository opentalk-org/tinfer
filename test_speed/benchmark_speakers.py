from dataclasses import dataclass
from pathlib import Path
from typing import Protocol
import json
import shutil

import torch

from test_speed.benchmark_data import ReferenceDuration, select_names
from test_speed.benchmark_inference import (
    archive_wav_names,
    embed_references,
    extract_selected,
    measure_reference_durations,
)


class VectorVoiceModel(Protocol):
    def load_voice_from_vector(
        self,
        voice_id: str,
        voice_vector: torch.Tensor,
    ) -> None: ...


class ArchiveVoiceModel(Protocol):
    def load_voice_from_audio(
        self,
        voice_id: str,
        audio_path: str,
        sample_rate: int | None = None,
    ) -> None: ...

    def get_voice(self, voice_id: str) -> torch.Tensor: ...


@dataclass(frozen=True)
class ArchiveVoiceSource:
    archive_path: Path


@dataclass(frozen=True)
class VectorVoiceSource:
    voices_dir: Path
    metadata_path: Path
    metadata_group: str


@dataclass(frozen=True)
class SingleVectorVoiceSource:
    voice_path: Path


@dataclass(frozen=True)
class TensorVoiceSource:
    voice_path: Path


@dataclass(frozen=True)
class TensorVoiceDirectorySource:
    voices_dir: Path


@dataclass(frozen=True)
class PreparedVoices:
    voice_ids: list[str]
    reference_durations: list[ReferenceDuration]
    source_names: list[str]


@dataclass(frozen=True)
class ReferenceMetadata:
    source_name: str
    duration_seconds: float


def prepare_archive_voices(
    model: ArchiveVoiceModel,
    source: ArchiveVoiceSource,
    references_dir: Path,
    embeddings_dir: Path,
    count: int,
    seed: int,
) -> PreparedVoices:
    available_names = archive_wav_names(source.archive_path)
    selected_names = select_names(available_names, count, seed)
    reference_paths = extract_selected(
        source.archive_path,
        selected_names,
        references_dir,
    )
    reference_durations = measure_reference_durations(reference_paths)
    voice_ids = embed_references(model, reference_paths, embeddings_dir)
    return PreparedVoices(voice_ids, reference_durations, selected_names)


def _read_reference_metadata(
    source: VectorVoiceSource,
) -> dict[str, ReferenceMetadata]:
    document = json.loads(source.metadata_path.read_text(encoding="utf-8"))
    entries = document[source.metadata_group]
    return {
        Path(entry["file"]).stem: ReferenceMetadata(
            Path(entry["file"]).name,
            float(entry["duration"]),
        )
        for entry in entries
    }


def prepare_vector_voices(
    model: VectorVoiceModel,
    source: VectorVoiceSource,
    output_dir: Path,
    count: int,
    seed: int,
) -> PreparedVoices:
    metadata = _read_reference_metadata(source)
    available_names = sorted(
        path.name
        for path in source.voices_dir.glob("*.pth")
        if path.stem in metadata
    )
    selected_names = select_names(available_names, count, seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    references = []
    source_names = []
    voice_ids = []
    for name in selected_names:
        path = source.voices_dir / name
        voice_id = path.stem
        voice_data = torch.load(path, map_location="cpu", weights_only=True)
        voice_vector = voice_data["tensor"]
        model.load_voice_from_vector(voice_id, voice_vector)
        shutil.copy2(path, output_dir / name)
        references.append(
            ReferenceDuration(voice_id, metadata[voice_id].duration_seconds)
        )
        source_names.append(metadata[voice_id].source_name)
        voice_ids.append(voice_id)
    return PreparedVoices(voice_ids, references, source_names)


def prepare_single_vector_voice(
    model: VectorVoiceModel,
    source: SingleVectorVoiceSource,
    output_dir: Path,
) -> PreparedVoices:
    voice_id = source.voice_path.stem
    voice_data = torch.load(
        source.voice_path,
        map_location="cpu",
        weights_only=True,
    )
    model.load_voice_from_vector(voice_id, voice_data["tensor"])
    output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source.voice_path, output_dir / source.voice_path.name)
    return PreparedVoices([voice_id], [], [source.voice_path.name])


def prepare_tensor_voice(
    model: VectorVoiceModel,
    source: TensorVoiceSource,
    output_dir: Path,
) -> PreparedVoices:
    voice_id = source.voice_path.stem
    voice_vector = torch.load(
        source.voice_path,
        map_location="cpu",
        weights_only=True,
    )
    model.load_voice_from_vector(voice_id, voice_vector)
    output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source.voice_path, output_dir / source.voice_path.name)
    return PreparedVoices([voice_id], [], [source.voice_path.name])


def prepare_tensor_voices(
    model: VectorVoiceModel,
    source: TensorVoiceDirectorySource,
    output_dir: Path,
    count: int,
    seed: int,
) -> PreparedVoices:
    available_names = sorted(path.name for path in source.voices_dir.glob("*.pth"))
    selected_names = select_names(available_names, count, seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    voice_ids = []
    for name in selected_names:
        path = source.voices_dir / name
        voice_id = path.stem
        voice_vector = torch.load(path, map_location="cpu", weights_only=True)
        model.load_voice_from_vector(voice_id, voice_vector)
        shutil.copy2(path, output_dir / name)
        voice_ids.append(voice_id)
    return PreparedVoices(voice_ids, [], selected_names)
