from dataclasses import dataclass
from pathlib import Path

from test_speed.benchmark_speakers import (
    ArchiveVoiceSource,
    SingleVectorVoiceSource,
    TensorVoiceSource,
    VectorVoiceSource,
)
from test_speed.benchmark_corpus import ENGLISH_PASSAGE, POLISH_PASSAGE


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
    voice_source: (
        ArchiveVoiceSource
        | VectorVoiceSource
        | SingleVectorVoiceSource
        | TensorVoiceSource
    )
    passage: str
    runtime_engine: str
    language: str


MAGDA_TARGET = BenchmarkTarget(
    name="magda",
    model_path=Path("/workspace/converted_models/magda/model.pth"),
    results_dir=ROOT / "test_speed/results",
    no_diffusion_results_dir=ROOT / "test_speed/results_no_diffusion",
    seed=20260710,
    voice_count=20,
    highlighted_voice_count=4,
    voice_source=ArchiveVoiceSource(ROOT / "test_speed/archive.zip"),
    passage=POLISH_PASSAGE,
    runtime_engine="tensorrt",
    language="pl",
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
    passage=POLISH_PASSAGE,
    runtime_engine="tensorrt",
    language="pl",
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
    passage=POLISH_PASSAGE,
    runtime_engine="tensorrt",
    language="pl",
)
LJSPEECH_VOICE = Path(
    "/workspace/converted_models/ljspeech/voices/ljspeech.pth"
)
VOKAN_TARGET = BenchmarkTarget(
    name="vokan",
    model_path=Path("/workspace/converted_models/vokan/model.pth"),
    results_dir=ROOT / "test_speed/results_vokan",
    no_diffusion_results_dir=ROOT / "test_speed/results_vokan_no_diffusion",
    seed=20260710,
    voice_count=1,
    highlighted_voice_count=1,
    voice_source=TensorVoiceSource(LJSPEECH_VOICE),
    passage=ENGLISH_PASSAGE,
    runtime_engine="torch",
    language="en-us",
)
LJSPEECH_TARGET = BenchmarkTarget(
    name="ljspeech",
    model_path=Path("/workspace/converted_models/ljspeech/model.pth"),
    results_dir=ROOT / "test_speed/results_ljspeech",
    no_diffusion_results_dir=ROOT / "test_speed/results_ljspeech_no_diffusion",
    seed=20260710,
    voice_count=1,
    highlighted_voice_count=1,
    voice_source=TensorVoiceSource(LJSPEECH_VOICE),
    passage=ENGLISH_PASSAGE,
    runtime_engine="torch",
    language="en-us",
)
TARGETS = (
    MAGDA_TARGET,
    AGNIESZKA_TARGET,
    OLAM_TARGET,
    VOKAN_TARGET,
    LJSPEECH_TARGET,
)
