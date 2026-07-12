from pathlib import Path
from zipfile import ZipFile
import random

import numpy as np
import torch

from test_speed.benchmark.benchmark_inference import load_model
from test_speed.investigations.padding.analysis import trace_duration_path
from test_speed.investigations.padding.report import write_report


ROOT = Path(__file__).resolve().parents[3]
MODEL_PATH = Path("/workspace/converted_models/magda/model.pth")
ARCHIVE_PATH = ROOT / "test_speed/archive.zip"
OUTPUT_DIR = ROOT / "test_speed/results/evidence/padding"
TEXT = (
    "Rankiem mieszkańcy kamienicy spotkali się na dziedzińcu, aby posadzić "
    "zioła, naprawić drewnianą ławkę i zawiesić kolorowe lampki."
)
PADDING_COUNTS = (0, 5, 10, 25, 50, 100, 200)


def _extract_reference() -> Path:
    output = OUTPUT_DIR / "reference.wav"
    output.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(ARCHIVE_PATH) as archive:
        name = sorted(
            item
            for item in archive.namelist()
            if item.endswith(".wav") and "__MACOSX" not in Path(item).parts
        )[0]
        output.write_bytes(archive.read(name))
    return output


def _real_tokens(model: object) -> tuple[str, list[int]]:
    phonemizer = model._get_phonemizer("pl")
    phonemes, _ = phonemizer.process_text_with_original_spans(TEXT)
    return phonemes, [0, *phonemizer.tokenize(phonemes)]


def main() -> None:
    random.seed(20260710)
    np.random.seed(20260710)
    torch.manual_seed(20260710)
    torch.cuda.manual_seed_all(20260710)
    model = load_model(MODEL_PATH, "cuda", "torch")
    for module in model._model.values():
        if isinstance(module, torch.nn.Module):
            module.eval()
    reference = _extract_reference()
    model.load_voice_from_audio("magda", str(reference))
    style = model.get_voice("magda").detach().reshape(1, 256)
    phonemes, tokens = _real_tokens(model)
    real_length = len(tokens)
    assert real_length + max(PADDING_COUNTS) <= 512
    traces = {}
    for padding in PADDING_COUNTS:
        padded = torch.tensor(
            [[*tokens, *([0] * padding)]],
            dtype=torch.long,
            device="cuda",
        )
        traces[padding] = trace_duration_path(
            model._model,
            padded,
            real_length,
            style,
        )
    write_report(
        OUTPUT_DIR,
        traces,
        real_length,
        {
            "model": str(MODEL_PATH),
            "reference": str(reference),
            "text": TEXT,
            "phonemes": phonemes,
            "tokens_with_start": tokens,
            "real_length": real_length,
            "padding_counts": list(PADDING_COUNTS),
            "eval_mode": True,
            "same_style_vector": True,
            "true_mask_preserved": True,
        },
    )
    print(f"Duration-padding report: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
