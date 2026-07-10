from __future__ import annotations

import argparse
import asyncio
import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from time import monotonic

import numpy as np
import torch

from tinfer.config.engine_config import StreamingTTSConfig
from tinfer.core.async_engine import AsyncStreamingTTS
from tinfer.core.engine import StreamingTTS
from tinfer.core.request import AlignmentType


MODEL_ID = "styletts2_trt_magda"
MODEL_PATH = Path("/workspace/converted_models/magda/model.pth")
VOICE_ID = "magda_001"
VOICES_FOLDER = Path("/workspace/converted_models/magda/voices")
OUT_DIR = Path("/workspace/tinfer/benchmark_outputs/scheduled_first_byte")

TEXT = (
    "W tym teście mierzymy opóźnienie do pierwszego fragmentu audio dla dłuższego tekstu, "
    "który przypomina realne użycie systemu syntezy mowy. Użytkownik może wysłać cały akapit, "
    "a serwer powinien szybko rozpocząć generowanie pierwszej odpowiedzi, zanim zakończy pracę nad całością. "
    "Dlatego pierwszy fragment ma większy limit długości niż w domyślnej konfiguracji i zawiera kilka zdań. "
    "Pomiar obejmuje kolejkowanie, grupowanie żądań, inferencję modelu oraz przekazanie pierwszych próbek audio "
    "z powrotem do strumienia klienta. Taki scenariusz lepiej pokazuje zachowanie systemu przy wielu jednoczesnych użytkownikach."
)


@dataclass
class Row:
    users: int
    repeats: int
    samples: int
    mean_ms: float
    p50_ms: float
    p90_ms: float
    p95_ms: float
    min_ms: float
    max_ms: float
    std_ms: float
    text_chars: int
    first_chunk_chars: int


async def first_byte(async_tts: AsyncStreamingTTS, params: dict, repeat: int, user: int) -> float:
    start = monotonic()
    async for chunk in async_tts.generate(MODEL_ID, VOICE_ID, f"Próba {repeat + 1}, użytkownik {user + 1}. {TEXT}", params):
        if chunk.error:
            raise RuntimeError(chunk.error)
        if len(chunk.audio) == 0:
            raise RuntimeError("empty audio chunk")
        return monotonic() - start
    raise RuntimeError("no audio chunk")


async def measure(async_tts: AsyncStreamingTTS, users: int, repeats: int, params: dict, first_chunk_chars: int) -> Row:
    values = []
    for repeat in range(repeats):
        values.extend(
            await asyncio.gather(*(first_byte(async_tts, params, repeat, user) for user in range(users)))
        )
        await asyncio.sleep(0.15)
    arr = np.asarray(values, dtype=np.float64) * 1000.0
    return Row(
        users=users,
        repeats=repeats,
        samples=len(arr),
        mean_ms=float(np.mean(arr)),
        p50_ms=float(np.percentile(arr, 50)),
        p90_ms=float(np.percentile(arr, 90)),
        p95_ms=float(np.percentile(arr, 95)),
        min_ms=float(np.min(arr)),
        max_ms=float(np.max(arr)),
        std_ms=float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
        text_chars=len(TEXT),
        first_chunk_chars=first_chunk_chars,
    )


async def run(args: argparse.Namespace) -> None:
    runtime = torch.load(MODEL_PATH, map_location="cpu", weights_only=True).get("runtime_config") or {}
    if runtime.get("engine") != "tensorrt":
        raise RuntimeError(f"expected TensorRT engine, got {runtime.get('engine')!r}")

    params = {
        "alignment_type": AlignmentType.NONE,
        "chunk_length_schedule": [args.first_chunk_chars] * 4,
        "max_phoneme_tokens": args.max_phoneme_tokens,
        "tts_params": {"diffusion_steps": 10, "embedding_scale": 1.0},
    }
    config = StreamingTTSConfig(
        default_batch_size=max(args.users),
        compile_models=False,
        default_alignment_type=AlignmentType.NONE,
        default_chunk_schedule=[args.first_chunk_chars] * 4,
    )
    tts = StreamingTTS(config)
    rows = []
    try:
        print("runtime_engine", runtime.get("engine"), flush=True)
        print("tensorrt_components", runtime.get("tensorrt", {}).get("components"), flush=True)
        print("first_chunk_chars", args.first_chunk_chars, flush=True)
        print("max_phoneme_tokens", args.max_phoneme_tokens, flush=True)
        tts.load_model(MODEL_ID, str(MODEL_PATH), voices_folder=str(VOICES_FOLDER))
        tts.warmup([MODEL_ID], [VOICE_ID], num_warmup_tasks=min(4, max(args.users)))
        async_tts = AsyncStreamingTTS(tts)
        await first_byte(async_tts, params, -1, 0)
        for users in args.users:
            row = await measure(async_tts, users, args.repeats, params, args.first_chunk_chars)
            rows.append(row)
            print(f"users={users} mean={row.mean_ms:.1f}ms p95={row.p95_ms:.1f}ms max={row.max_ms:.1f}ms", flush=True)
    finally:
        tts.stop()
        await asyncio.sleep(0.2)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUT_DIR / "scheduled_first_byte.csv"
    json_path = OUT_DIR / "scheduled_first_byte.json"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(rows[0]).keys()))
        writer.writeheader()
        writer.writerows(asdict(row) for row in rows)
    json_path.write_text(json.dumps([asdict(row) for row in rows], indent=2), encoding="utf-8")
    print("csv", csv_path, flush=True)
    print("json", json_path, flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--users", type=int, nargs="+", default=[1, 2, 4, 8, 16])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--first-chunk-chars", type=int, default=420)
    parser.add_argument("--max-phoneme-tokens", type=int, default=512)
    return parser.parse_args()


if __name__ == "__main__":
    asyncio.run(run(parse_args()))
