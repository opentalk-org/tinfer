from __future__ import annotations

import argparse
import asyncio
import csv
import json
import subprocess
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from time import monotonic

import numpy as np

from tinfer.config.engine_config import StreamingTTSConfig
from tinfer.core.async_engine import AsyncStreamingTTS
from tinfer.core.engine import StreamingTTS
from tinfer.core.request import AlignmentType


BASE_DIR = Path("/workspace")
MODEL_PATH = BASE_DIR / "converted_models" / "magda" / "model.pth"
VOICE_ID = "magda_001"
VOICES_FOLDER = BASE_DIR / "converted_models" / "magda" / "voices"
OUT_DIR = BASE_DIR / "tinfer" / "benchmark_outputs"

LATENCY_TEXT = "To jest krotki tekst testowy do pomiaru pierwszego fragmentu audio."
THROUGHPUT_TEXT = "To jest tekst testowy do pomiaru przepustowosci syntezy mowy dla wielu uzytkownikow."


@dataclass
class VramRow:
    batch_size: int
    model_count: int
    users: int
    resident_memory_mib: int
    peak_memory_mib: int
    latency_mean_ms: float
    latency_std_ms: float
    latency_min_ms: float
    latency_max_ms: float
    latency_samples: int
    throughput_requests_per_sec: float
    throughput_wall_sec: float


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
    return int(result.stdout.strip().splitlines()[0].strip())


class VramSampler:
    def __init__(self, interval_sec: float = 0.01) -> None:
        self.interval_sec = interval_sec
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._samples: list[int] = []
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def reset(self) -> None:
        with self._lock:
            self._samples = [gpu_memory_mib()]

    def peak(self) -> int:
        with self._lock:
            return max(self._samples) if self._samples else gpu_memory_mib()

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                value = gpu_memory_mib()
            except Exception:
                time.sleep(self.interval_sec)
                continue
            with self._lock:
                self._samples.append(value)
            time.sleep(self.interval_sec)


async def first_chunk_latency(async_tts: AsyncStreamingTTS, model_id: str) -> float | None:
    start = monotonic()
    async for _chunk in async_tts.generate(
        model_id,
        VOICE_ID,
        LATENCY_TEXT,
        {"alignment_type": AlignmentType.NONE},
    ):
        return monotonic() - start
    return None


async def full_request(async_tts: AsyncStreamingTTS, model_id: str) -> float:
    start = monotonic()
    async for _chunk in async_tts.generate(
        model_id,
        VOICE_ID,
        THROUGHPUT_TEXT,
        {"alignment_type": AlignmentType.NONE},
    ):
        pass
    return monotonic() - start


async def measure_users(
    async_tts: AsyncStreamingTTS,
    model_ids: list[str],
    users: int,
    repeats: int,
) -> tuple[np.ndarray, float]:
    latency_values = []
    for repeat in range(repeats):
        results = await asyncio.gather(
            *(
                first_chunk_latency(async_tts, model_ids[(i + repeat * users) % len(model_ids)])
                for i in range(users)
            )
        )
        latency_values.extend(v for v in results if v is not None)

    throughput_start = monotonic()
    await asyncio.gather(*(full_request(async_tts, model_ids[i % len(model_ids)]) for i in range(users)))
    throughput_wall_sec = monotonic() - throughput_start

    if not latency_values:
        raise RuntimeError(f"No first-chunk latencies measured for users={users}")
    return np.asarray(latency_values, dtype=float) * 1000.0, throughput_wall_sec


async def run_scenario(args: argparse.Namespace, model_count: int) -> list[VramRow]:
    model_ids = [f"styletts2_{i}" for i in range(model_count)]
    config = StreamingTTSConfig(
        default_batch_size=args.current_batch_size,
        compile_models=False,
        default_alignment_type=AlignmentType.NONE,
    )

    sampler = VramSampler(args.sample_interval_ms / 1000.0)
    sampler.start()
    tts = StreamingTTS(config)
    try:
        for model_id in model_ids:
            tts.load_model(model_id, str(MODEL_PATH), voices_folder=str(VOICES_FOLDER))

        async_tts = AsyncStreamingTTS(tts)
        sampler.reset()
        for model_id in model_ids:
            await first_chunk_latency(async_tts, model_id)
        resident_memory_mib = max(sampler.peak(), gpu_memory_mib())

        rows: list[VramRow] = []
        for users in args.users:
            sampler.reset()
            lat_ms, throughput_wall_sec = await measure_users(async_tts, model_ids, users, args.repeats)
            peak_memory_mib = sampler.peak()
            row = VramRow(
                batch_size=args.current_batch_size,
                model_count=model_count,
                users=users,
                resident_memory_mib=resident_memory_mib,
                peak_memory_mib=peak_memory_mib,
                latency_mean_ms=float(np.mean(lat_ms)),
                latency_std_ms=float(np.std(lat_ms, ddof=1)) if len(lat_ms) > 1 else 0.0,
                latency_min_ms=float(np.min(lat_ms)),
                latency_max_ms=float(np.max(lat_ms)),
                latency_samples=len(lat_ms),
                throughput_requests_per_sec=float(users / throughput_wall_sec),
                throughput_wall_sec=float(throughput_wall_sec),
            )
            rows.append(row)
            print(
                f"batch={args.current_batch_size} models={model_count} users={users} "
                f"peak={row.peak_memory_mib}MiB latency_mean={row.latency_mean_ms:.1f}ms "
                f"throughput={row.throughput_requests_per_sec:.2f}req/s",
                flush=True,
            )
        return rows
    finally:
        tts.stop()
        sampler.stop()
        await asyncio.sleep(0.2)


def write_outputs(rows: list[VramRow], out_prefix: str) -> tuple[Path, Path]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUT_DIR / f"{out_prefix}.csv"
    json_path = OUT_DIR / f"{out_prefix}.json"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(rows[0]).keys()))
        writer.writeheader()
        writer.writerows(asdict(row) for row in rows)
    json_path.write_text(json.dumps([asdict(row) for row in rows], indent=2), encoding="utf-8")
    return csv_path, json_path


async def run(args: argparse.Namespace) -> None:
    all_rows: list[VramRow] = []
    batch_sizes = args.batch_sizes if args.batch_sizes is not None else [args.batch_size]
    for batch_size in batch_sizes:
        args.current_batch_size = batch_size
        for model_count in args.model_counts:
            print(f"\n=== batch {batch_size}, {model_count} loaded model(s) ===", flush=True)
            rows = await run_scenario(args, model_count)
            all_rows.extend(rows)

    csv_path, json_path = write_outputs(all_rows, args.out_prefix)
    print(f"\nCSV: {csv_path}")
    print(f"JSON: {json_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark GPU memory with one or more loaded StyleTTS2 models.")
    parser.add_argument("--model-counts", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--users", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32, 64])
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--batch-sizes", type=int, nargs="+")
    parser.add_argument("--sample-interval-ms", type=float, default=10.0)
    parser.add_argument("--out-prefix", default="trt_dynamic_fp16_b64_t512_vram_models")
    return parser.parse_args()


def main() -> None:
    asyncio.run(run(parse_args()))


if __name__ == "__main__":
    main()
