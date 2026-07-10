from dataclasses import dataclass
from pathlib import Path

from test_speed.benchmark_speakers import (
    ArchiveVoiceSource,
    SingleVectorVoiceSource,
    VectorVoiceSource,
)


ROOT = Path(__file__).resolve().parents[1]
AGNIESZKA_ROOT = Path("/workspace/converted_models/agnieszka")


@dataclass(frozen=True)
class BenchmarkTarget:
    name: str
    model_path: Path
    results_dir: Path
    no_diffusion_results_dir: Path
    seed: int
    voice_count: int
    highlighted_voice_count: int
    voice_source: ArchiveVoiceSource | VectorVoiceSource | SingleVectorVoiceSource


MAGDA_TARGET = BenchmarkTarget(
    name="magda",
    model_path=Path("/workspace/converted_models/magda/model.pth"),
    results_dir=ROOT / "test_speed/results",
    no_diffusion_results_dir=ROOT / "test_speed/results_no_diffusion",
    seed=20260710,
    voice_count=20,
    highlighted_voice_count=4,
    voice_source=ArchiveVoiceSource(ROOT / "test_speed/archive.zip"),
)
AGNIESZKA_TARGET = BenchmarkTarget(
    name="agnieszka",
    model_path=AGNIESZKA_ROOT / "model.pth",
    results_dir=ROOT / "test_speed/results_agnieszka",
    no_diffusion_results_dir=ROOT / "test_speed/results_agnieszka_no_diffusion",
    seed=20260710,
    voice_count=20,
    highlighted_voice_count=4,
    voice_source=VectorVoiceSource(
        AGNIESZKA_ROOT / "voices",
        Path(
            "/workspace/other/remote_model_sources/agnieszka/"
            "agnieszka_multispeaker_emb.json"
        ),
        "agnieszka-best",
    ),
)
OLAM_TARGET = BenchmarkTarget(
    name="olam",
    model_path=Path("/workspace/converted_models/olam/model.pth"),
    results_dir=ROOT / "test_speed/results_olam",
    no_diffusion_results_dir=ROOT / "test_speed/results_olam_no_diffusion",
    seed=20260710,
    voice_count=1,
    highlighted_voice_count=1,
    voice_source=SingleVectorVoiceSource(
        Path("/workspace/converted_models/olam/voices/any.pth")
    ),
)
TARGETS = (MAGDA_TARGET, AGNIESZKA_TARGET, OLAM_TARGET)
