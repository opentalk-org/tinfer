from argparse import ArgumentParser
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import assert_never
import json
import logging
import random
import shutil

import numpy as np
import torch

import tinfer.models.impl.styletts2.model.model as styletts2_model_module
from tinfer.support.observability import setup_json_logs

from test_speed.benchmark_data import (
    PhonemeMetric,
    RequestMetric,
    SynthesisProfile,
    TextInput,
    select_names,
)
from test_speed.benchmark_corpus import POLISH_PASSAGE, build_phoneme_grid
from test_speed.benchmark_config import (
    MAGDA_TARGET,
    TARGETS,
    BenchmarkTarget,
)
from test_speed.benchmark_inference import (
    load_model,
    synthesize_all,
)
from test_speed.benchmark_reporting import shared_histogram_edges, write_reports
from test_speed.benchmark_speakers import (
    ArchiveVoiceSource,
    PreparedVoices,
    SingleVectorVoiceSource,
    VectorVoiceSource,
    prepare_archive_voices,
    prepare_single_vector_voice,
    prepare_vector_voices,
)
from test_speed.benchmark_style import measure_style_norms


CONFIG = MAGDA_TARGET
PROFILES = [
    SynthesisProfile("diffusion", CONFIG.results_dir, True),
    SynthesisProfile(
        "no_diffusion",
        CONFIG.no_diffusion_results_dir,
        False,
    ),
]
HISTOGRAM_BIN_WIDTH = 0.25


@dataclass(frozen=True)
class ProfileMetrics:
    profile: SynthesisProfile
    requests: list[RequestMetric]
    phonemes: list[PhonemeMetric]


def preserve_requested_speed(speed: float, _token_count: int) -> float:
    return speed


def disable_speed_correction() -> None:
    styletts2_model_module.baseline_speed_corrected_for_request = (
        preserve_requested_speed
    )


def configure_progress_output() -> None:
    setup_json_logs(level=logging.WARNING, force=True)


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def _prepare_results(results_dir: Path) -> None:
    if results_dir.exists():
        shutil.rmtree(results_dir)
    for name in ("references", "embeddings", "audio", "metrics", "summary"):
        (results_dir / name).mkdir(parents=True, exist_ok=True)


def copy_profile_inputs(source_dir: Path, destination_dir: Path) -> None:
    for name in ("references", "embeddings"):
        shutil.copytree(
            source_dir / name,
            destination_dir / name,
            dirs_exist_ok=True,
        )


def select_targets(selection: str) -> list[BenchmarkTarget]:
    if selection == "all":
        return list(TARGETS)
    matches = [target for target in TARGETS if target.name == selection]
    if not matches:
        raise ValueError(f"Unknown benchmark target: {selection}")
    return matches


def _profiles_for(target: BenchmarkTarget) -> list[SynthesisProfile]:
    return [
        SynthesisProfile("diffusion", target.results_dir, True),
        SynthesisProfile(
            "no_diffusion",
            target.no_diffusion_results_dir,
            False,
        ),
    ]


def _prepare_voices(
    target: BenchmarkTarget,
    model: object,
    primary_results_dir: Path,
) -> PreparedVoices:
    source = target.voice_source
    if isinstance(source, ArchiveVoiceSource):
        return prepare_archive_voices(
            model,
            source,
            primary_results_dir / "references",
            primary_results_dir / "embeddings",
            target.voice_count,
            target.seed,
        )
    if isinstance(source, VectorVoiceSource):
        return prepare_vector_voices(
            model,
            source,
            primary_results_dir / "embeddings",
            target.voice_count,
            target.seed,
        )
    if isinstance(source, SingleVectorVoiceSource):
        return prepare_single_vector_voice(
            model,
            source,
            primary_results_dir / "embeddings",
        )
    assert_never(source)


def _write_manifest(
    config: BenchmarkTarget,
    profile: SynthesisProfile,
    selected_names: list[str],
    highlighted_voice_ids: list[str],
    text_inputs: list[TextInput],
) -> Path:
    manifest = {
        "speaker": config.name,
        "profile": profile.name,
        "use_diffusion": profile.use_diffusion,
        "speed_correction": False,
        "seed": config.seed,
        "voice_source": str(config.voice_source),
        "model_path": str(config.model_path),
        "selected_reference_wavs": selected_names,
        "highlighted_voice_ids": highlighted_voice_ids,
        "texts": [asdict(text_input) for text_input in text_inputs],
    }
    path = profile.results_dir / "manifest.json"
    path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return path


def _validate_profile_metrics(
    config: BenchmarkTarget,
    metrics: ProfileMetrics,
    text_inputs: list[TextInput],
) -> None:
    expected_requests = config.voice_count * len(text_inputs)
    assert len(metrics.requests) == expected_requests, (
        f"{metrics.profile.name} produced {len(metrics.requests)} requests"
    )
    requests_by_voice = Counter(row.voice_id for row in metrics.requests)
    assert len(requests_by_voice) == config.voice_count
    assert set(requests_by_voice.values()) == {len(text_inputs)}


def _run_target(config: BenchmarkTarget) -> None:
    _seed_everything(config.seed)
    profiles = _profiles_for(config)
    for profile in profiles:
        _prepare_results(profile.results_dir)
    model = load_model(config.model_path, "cuda")
    text_inputs = build_phoneme_grid(
        model,
        POLISH_PASSAGE,
        point_count=48,
        max_tokens=511,
    )
    prepared_voices = _prepare_voices(
        config,
        model,
        profiles[0].results_dir,
    )
    style_norms = measure_style_norms(model, prepared_voices.voice_ids)
    copy_profile_inputs(profiles[0].results_dir, profiles[1].results_dir)
    highlighted_voice_ids = select_names(
        prepared_voices.voice_ids,
        config.highlighted_voice_count,
        config.seed + 1,
    )
    profile_metrics = []
    for profile in profiles:
        requests, phonemes = synthesize_all(
            model,
            prepared_voices.voice_ids,
            text_inputs,
            profile.results_dir / "audio",
            profile.use_diffusion,
        )
        metrics = ProfileMetrics(profile, requests, phonemes)
        _validate_profile_metrics(config, metrics, text_inputs)
        profile_metrics.append(metrics)

    histogram_edges = shared_histogram_edges(
        *[
            [row.phonemes_per_second for row in metrics.requests]
            for metrics in profile_metrics
        ],
        bin_width=HISTOGRAM_BIN_WIDTH,
    )
    for metrics in profile_metrics:
        index_path = write_reports(
            metrics.profile.results_dir,
            metrics.requests,
            metrics.phonemes,
            prepared_voices.reference_durations,
            style_norms,
            highlighted_voice_ids,
            histogram_edges,
        )
        manifest_path = _write_manifest(
            config,
            metrics.profile,
            prepared_voices.source_names,
            highlighted_voice_ids,
            text_inputs,
        )
        print(f"{metrics.profile.name} summary: {index_path}")
        print(f"{metrics.profile.name} manifest: {manifest_path}")


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument(
        "target",
        choices=("all", "magda", "agnieszka", "olam"),
        default="all",
        nargs="?",
    )
    arguments = parser.parse_args()
    configure_progress_output()
    disable_speed_correction()
    for target in select_targets(arguments.target):
        _run_target(target)


if __name__ == "__main__":
    main()
