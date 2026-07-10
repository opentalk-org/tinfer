from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import torch

from tinfer.core.request import AlignmentType
from tinfer.models.impl.styletts2.model.model import StyleTTS2


MODEL_PATH = Path("/workspace/converted_models/magda/model.pth")
VOICES_FOLDER = Path("/workspace/converted_models/magda/voices")
VOICE_ID = "magda_001"
TEXT = "To jest tekst testowy do pomiaru pamieci modelu."


def gpu_memory_mib() -> int:
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=memory.used",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=2.0,
    )
    return int(result.stdout.strip().splitlines()[0])


def snapshot(stage: str) -> dict[str, float | int | str]:
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return {
        "stage": stage,
        "gpu_memory_mib": gpu_memory_mib(),
        "torch_allocated_mib": torch.cuda.memory_allocated() // (1 << 20),
        "torch_reserved_mib": torch.cuda.memory_reserved() // (1 << 20),
        "torch_max_allocated_mib": torch.cuda.max_memory_allocated() // (1 << 20),
        "torch_max_reserved_mib": torch.cuda.max_memory_reserved() // (1 << 20),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure direct StyleTTS2 VRAM stages in one clean process.")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--scenario", default="unknown")
    args = parser.parse_args()

    rows = [snapshot("process_start")]
    model = StyleTTS2(device="cuda")
    rows.append(snapshot("after_init"))
    model.load(str(MODEL_PATH), voices_folder=str(VOICES_FOLDER), device="cuda", compile_model=False)
    rows.append(snapshot("after_load"))

    contexts = [{"voice_id": VOICE_ID} for _ in range(args.batch_size)]
    params = [{"use_diffusion": True, "diffusion_steps": 10, "embedding_scale": 1.0} for _ in range(args.batch_size)]
    metadata = [
        {
            "alignment_type": AlignmentType.NONE,
            "text_span": (0, len(TEXT)),
            "text": TEXT,
            "request_id": str(i),
            "chunk_index": 0,
        }
        for i in range(args.batch_size)
    ]
    texts = [TEXT] * args.batch_size
    error = None
    try:
        model.generate_batch(texts, contexts, params, metadata)
        rows.append(snapshot("after_generate"))
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        rows.append(snapshot("after_error"))

    print(json.dumps({"scenario": args.scenario, "batch_size": args.batch_size, "error": error, "rows": rows}, indent=2))
    if error is not None:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
