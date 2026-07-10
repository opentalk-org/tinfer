from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
import json
import logging
import random
import shutil

import numpy as np
import torch

import tinfer.models.impl.styletts2.model.model as styletts2_model_module
from tinfer.support.observability import setup_json_logs

from test_speed.benchmark_data import (
    BenchmarkConfig,
    PhonemeMetric,
    RequestMetric,
    SynthesisProfile,
    TextInput,
    select_names,
)
from test_speed.benchmark_corpus import POLISH_PASSAGE, build_phoneme_grid
from test_speed.benchmark_inference import (
    archive_wav_names,
    embed_references,
    extract_selected,
    load_model,
    synthesize_all,
)
from test_speed.benchmark_reporting import shared_histogram_edges, write_reports


ROOT = Path(__file__).resolve().parents[1]
CONFIG = BenchmarkConfig(
    archive_path=ROOT / "test_speed/archive.zip",
    model_path=Path("/workspace/converted_models/magda/model.pth"),
    results_dir=ROOT / "test_speed/results",
    seed=20260710,
    voice_count=20,
    highlighted_voice_count=4,
)
PROFILES = [
    SynthesisProfile("diffusion", CONFIG.results_dir, True),
    SynthesisProfile(
        "no_diffusion",
        ROOT / "test_speed/results_no_diffusion",
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


def _write_manifest(
    config: BenchmarkConfig,
    profile: SynthesisProfile,
    selected_names: list[str],
    highlighted_voice_ids: list[str],
    text_inputs: list[TextInput],
) -> Path:
    manifest = {
        "profile": profile.name,
        "use_diffusion": profile.use_diffusion,
        "speed_correction": False,
        "seed": config.seed,
        "archive_path": str(config.archive_path),
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
    config: BenchmarkConfig,
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


def main() -> None:
    configure_progress_output()
    disable_speed_correction()
    _seed_everything(CONFIG.seed)
    for profile in PROFILES:
        _prepare_results(profile.results_dir)
    available_names = archive_wav_names(CONFIG.archive_path)
    selected_names = select_names(
        available_names,
        CONFIG.voice_count,
        CONFIG.seed,
    )
    reference_paths = extract_selected(
        CONFIG.archive_path,
        selected_names,
        PROFILES[0].results_dir / "references",
    )
    model = load_model(CONFIG.model_path, "cuda")
    text_inputs = build_phoneme_grid(
        model,
        POLISH_PASSAGE,
        point_count=48,
        max_tokens=511,
    )
    voice_ids = embed_references(
        model,
        reference_paths,
        PROFILES[0].results_dir / "embeddings",
    )
    copy_profile_inputs(PROFILES[0].results_dir, PROFILES[1].results_dir)
    highlighted_voice_ids = select_names(
        voice_ids,
        CONFIG.highlighted_voice_count,
        CONFIG.seed + 1,
    )
    profile_metrics = []
    for profile in PROFILES:
        requests, phonemes = synthesize_all(
            model,
            voice_ids,
            text_inputs,
            profile.results_dir / "audio",
            profile.use_diffusion,
        )
        metrics = ProfileMetrics(profile, requests, phonemes)
        _validate_profile_metrics(CONFIG, metrics, text_inputs)
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
            highlighted_voice_ids,
            histogram_edges,
        )
        manifest_path = _write_manifest(
            CONFIG,
            metrics.profile,
            selected_names,
            highlighted_voice_ids,
            text_inputs,
        )
        print(f"{metrics.profile.name} summary: {index_path}")
        print(f"{metrics.profile.name} manifest: {manifest_path}")


if __name__ == "__main__":
    main()
