from dataclasses import asdict
from pathlib import Path
import json
import logging
import random
import shutil

import numpy as np
import torch

from tinfer.support.observability import setup_json_logs

from test_speed.benchmark_data import (
    POLISH_INPUTS,
    BenchmarkConfig,
    select_names,
)
from test_speed.benchmark_inference import (
    archive_wav_names,
    embed_references,
    extract_selected,
    load_model,
    synthesize_all,
)
from test_speed.benchmark_reporting import write_reports


ROOT = Path(__file__).resolve().parents[1]
CONFIG = BenchmarkConfig(
    archive_path=ROOT / "test_speed/archive.zip",
    model_path=Path("/workspace/converted_models/magda/model.pth"),
    results_dir=ROOT / "test_speed/results",
    seed=20260710,
    voice_count=20,
    highlighted_voice_count=4,
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


def _write_manifest(
    config: BenchmarkConfig,
    selected_names: list[str],
    highlighted_voice_ids: list[str],
) -> Path:
    manifest = {
        "seed": config.seed,
        "archive_path": str(config.archive_path),
        "model_path": str(config.model_path),
        "selected_reference_wavs": selected_names,
        "highlighted_voice_ids": highlighted_voice_ids,
        "texts": [asdict(text_input) for text_input in POLISH_INPUTS],
    }
    path = config.results_dir / "manifest.json"
    path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return path


def main() -> None:
    configure_progress_output()
    _seed_everything(CONFIG.seed)
    _prepare_results(CONFIG.results_dir)
    available_names = archive_wav_names(CONFIG.archive_path)
    selected_names = select_names(
        available_names,
        CONFIG.voice_count,
        CONFIG.seed,
    )
    reference_paths = extract_selected(
        CONFIG.archive_path,
        selected_names,
        CONFIG.results_dir / "references",
    )
    model = load_model(CONFIG.model_path, "cuda")
    voice_ids = embed_references(
        model,
        reference_paths,
        CONFIG.results_dir / "embeddings",
    )
    highlighted_voice_ids = select_names(
        voice_ids,
        CONFIG.highlighted_voice_count,
        CONFIG.seed + 1,
    )
    request_metrics, phoneme_metrics = synthesize_all(
        model,
        voice_ids,
        POLISH_INPUTS,
        CONFIG.results_dir / "audio",
    )
    index_path = write_reports(
        CONFIG.results_dir,
        request_metrics,
        phoneme_metrics,
        highlighted_voice_ids,
    )
    manifest_path = _write_manifest(CONFIG, selected_names, highlighted_voice_ids)
    print(f"Summary: {index_path}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
