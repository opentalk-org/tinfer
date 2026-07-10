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


BASE_DIR = Path("/workspace")
MODEL_ID = "styletts2_trt_magda"
MODEL_PATH = BASE_DIR / "converted_models" / "magda" / "model.pth"
VOICE_ID = "magda_001"
VOICES_FOLDER = BASE_DIR / "converted_models" / "magda" / "voices"
OUT_DIR = BASE_DIR / "tinfer" / "benchmark_outputs" / "tensorrt_first_byte_long_text"

LONG_TEXT = (
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
    first_byte_mean_ms: float
    first_byte_p50_ms: float
    first_byte_p90_ms: float
    first_byte_p95_ms: float
    first_byte_min_ms: float
    first_byte_max_ms: float
    first_byte_std_ms: float
    wall_mean_ms: float
    text_chars: int
    first_chunk_max_chars: int


def build_params(first_chunk_chars: int) -> dict:
    return {
        "alignment_type": AlignmentType.NONE,
        "chunk_length_schedule": [first_chunk_chars] * 4,
        "tts_params": {"diffusion_steps": 5, "embedding_scale": 1.0},
    }


async def first_byte(
    async_tts: AsyncStreamingTTS,
    *,
    user_index: int,
    repeat_index: int,
    params: dict,
) -> float:
    text = f"Użytkownik {user_index + 1}, próba {repeat_index + 1}. {LONG_TEXT}"
    start = monotonic()
    async for chunk in async_tts.generate(MODEL_ID, VOICE_ID, text, params):
        if getattr(chunk, "error", None):
            raise RuntimeError(chunk.error)
        if len(chunk.audio) <= 0:
            raise RuntimeError("received empty first audio chunk")
        return monotonic() - start
    raise RuntimeError("stream ended before first audio chunk")


async def measure(async_tts: AsyncStreamingTTS, *, users: int, repeats: int, params: dict) -> Row:
    all_latencies: list[float] = []
    wall_times: list[float] = []
    for repeat in range(repeats):
        wall_start = monotonic()
        latencies = await asyncio.gather(
            *(first_byte(async_tts, user_index=i, repeat_index=repeat, params=params) for i in range(users))
        )
        wall_times.append(monotonic() - wall_start)
        all_latencies.extend(latencies)
        await asyncio.sleep(0.15)

    arr = np.asarray(all_latencies, dtype=np.float64) * 1000.0
    wall = np.asarray(wall_times, dtype=np.float64) * 1000.0
    return Row(
        users=users,
        repeats=repeats,
        samples=len(arr),
        first_byte_mean_ms=float(np.mean(arr)),
        first_byte_p50_ms=float(np.percentile(arr, 50)),
        first_byte_p90_ms=float(np.percentile(arr, 90)),
        first_byte_p95_ms=float(np.percentile(arr, 95)),
        first_byte_min_ms=float(np.min(arr)),
        first_byte_max_ms=float(np.max(arr)),
        first_byte_std_ms=float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
        wall_mean_ms=float(np.mean(wall)),
        text_chars=len(LONG_TEXT),
        first_chunk_max_chars=params["chunk_length_schedule"][0],
    )


async def run(args: argparse.Namespace) -> None:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(MODEL_PATH)
    if not (VOICES_FOLDER / f"{VOICE_ID}.pth").exists():
        raise FileNotFoundError(VOICES_FOLDER / f"{VOICE_ID}.pth")

    saved = torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
    runtime = saved.get("runtime_config") or {}
    if runtime.get("engine") != "tensorrt":
        raise RuntimeError(f"expected TensorRT model, got runtime engine {runtime.get('engine')!r}")

    params = build_params(args.first_chunk_chars)
    print("runtime_engine", runtime.get("engine"), flush=True)
    print("tensorrt_components", runtime.get("tensorrt", {}).get("components"), flush=True)
    print("text_chars", len(LONG_TEXT), flush=True)
    print("first_chunk_max_chars", args.first_chunk_chars, flush=True)

    config = StreamingTTSConfig(
        default_batch_size=max(args.users),
        compile_models=False,
        default_alignment_type=AlignmentType.NONE,
        default_chunk_schedule=[args.first_chunk_chars] * 4,
        default_timeout_ms=80.0,
    )
    tts = StreamingTTS(config)
    rows: list[Row] = []
    try:
        print("loading_model", MODEL_PATH, flush=True)
        tts.load_model(MODEL_ID, str(MODEL_PATH), voices_folder=str(VOICES_FOLDER))
        print("warmup_start", flush=True)
        await tts.async_warmup([MODEL_ID], [VOICE_ID], num_warmup_tasks=min(4, max(args.users)))
        async_tts = AsyncStreamingTTS(tts)
        await first_byte(async_tts, user_index=0, repeat_index=-1, params=params)
        print("warmup_done", flush=True)

        for users in args.users:
            print(f"benchmark_users {users}", flush=True)
            row = await measure(async_tts, users=users, repeats=args.repeats, params=params)
            rows.append(row)
            print(
                f"users={users} mean={row.first_byte_mean_ms:.1f}ms "
                f"p50={row.first_byte_p50_ms:.1f}ms p95={row.first_byte_p95_ms:.1f}ms "
                f"min={row.first_byte_min_ms:.1f}ms max={row.first_byte_max_ms:.1f}ms",
                flush=True,
            )
    finally:
        tts.stop()
        await asyncio.sleep(0.2)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUT_DIR / "first_byte_latency_long_text.csv"
    json_path = OUT_DIR / "first_byte_latency_long_text.json"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(rows[0]).keys()))
        writer.writeheader()
        writer.writerows(asdict(row) for row in rows)
    json_path.write_text(json.dumps([asdict(row) for row in rows], indent=2, ensure_ascii=False), encoding="utf-8")
    print("csv", csv_path, flush=True)
    print("json", json_path, flush=True)
    print("rows_json", flush=True)
    print(json.dumps([asdict(row) for row in rows], indent=2, ensure_ascii=False), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark TensorRT first-byte latency with long text.")
    parser.add_argument("--users", type=int, nargs="+", default=[1, 2, 4, 8, 16])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--first-chunk-chars", type=int, default=120)
    return parser.parse_args()


if __name__ == "__main__":
    asyncio.run(run(parse_args()))
