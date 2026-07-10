from __future__ import annotations

import argparse
import asyncio
import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from time import monotonic

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from tinfer.config.engine_config import StreamingTTSConfig
from tinfer.core.async_engine import AsyncStreamingTTS
from tinfer.core.engine import StreamingTTS
from tinfer.core.request import AlignmentType


BASE_DIR = Path("/workspace")
MODEL_ID = "styletts2"
MODEL_PATH = BASE_DIR / "converted_models" / "magda" / "model.pth"
VOICE_ID = "magda_001"
VOICES_FOLDER = BASE_DIR / "converted_models" / "magda" / "voices"
OUT_DIR = BASE_DIR / "tinfer" / "benchmark_outputs"

LATENCY_TEXT = "To jest krótki tekst testowy do pomiaru pierwszego fragmentu audio."
THROUGHPUT_TEXT = "To jest tekst testowy do pomiaru przepustowości syntezy mowy dla wielu użytkowników."


@dataclass
class BenchmarkRow:
    users: int
    latency_mean_ms: float
    latency_std_ms: float
    latency_min_ms: float
    latency_max_ms: float
    latency_samples: int
    throughput_requests_per_sec: float
    throughput_audio_sec_per_sec: float
    throughput_wall_sec: float
    throughput_audio_sec: float


async def first_chunk_latency(async_tts: AsyncStreamingTTS) -> float | None:
    start = monotonic()
    async for _chunk in async_tts.generate(
        MODEL_ID,
        VOICE_ID,
        LATENCY_TEXT,
        {"alignment_type": AlignmentType.NONE},
    ):
        return monotonic() - start
    return None


async def full_request(async_tts: AsyncStreamingTTS) -> float:
    audio_samples = 0
    sample_rate = 24000
    async for chunk in async_tts.generate(
        MODEL_ID,
        VOICE_ID,
        THROUGHPUT_TEXT,
        {"alignment_type": AlignmentType.NONE},
    ):
        audio_samples += len(chunk.audio)
        sample_rate = chunk.sample_rate
    return audio_samples / sample_rate


async def measure_users(async_tts: AsyncStreamingTTS, users: int, repeats: int) -> BenchmarkRow:
    latency_values = []
    for _ in range(repeats):
        results = await asyncio.gather(*(first_chunk_latency(async_tts) for _ in range(users)))
        latency_values.extend(v for v in results if v is not None)

    if not latency_values:
        raise RuntimeError(f"No first-chunk latencies measured for users={users}")

    throughput_start = monotonic()
    audio_durations = await asyncio.gather(*(full_request(async_tts) for _ in range(users)))
    throughput_wall_sec = monotonic() - throughput_start
    audio_sec = float(sum(audio_durations))

    lat_ms = np.asarray(latency_values, dtype=float) * 1000.0
    return BenchmarkRow(
        users=users,
        latency_mean_ms=float(np.mean(lat_ms)),
        latency_std_ms=float(np.std(lat_ms, ddof=1)) if len(lat_ms) > 1 else 0.0,
        latency_min_ms=float(np.min(lat_ms)),
        latency_max_ms=float(np.max(lat_ms)),
        latency_samples=len(lat_ms),
        throughput_requests_per_sec=users / throughput_wall_sec,
        throughput_audio_sec_per_sec=audio_sec / throughput_wall_sec,
        throughput_wall_sec=throughput_wall_sec,
        throughput_audio_sec=audio_sec,
    )


def write_results(rows: list[BenchmarkRow], out_dir: Path) -> tuple[Path, Path]:
    csv_path = out_dir / "latency_throughput_results.csv"
    json_path = out_dir / "latency_throughput_results.json"

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(rows[0]).keys()))
        writer.writeheader()
        writer.writerows(asdict(row) for row in rows)

    json_path.write_text(
        json.dumps([asdict(row) for row in rows], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return csv_path, json_path


def plot_latency(rows: list[BenchmarkRow], out_dir: Path) -> Path:
    users = np.asarray([row.users for row in rows])
    mean = np.asarray([row.latency_mean_ms for row in rows])
    std = np.asarray([row.latency_std_ms for row in rows])
    min_v = np.asarray([row.latency_min_ms for row in rows])
    max_v = np.asarray([row.latency_max_ms for row in rows])

    fig, ax = plt.subplots(figsize=(11, 6.5))
    ax.plot(users, mean, marker="o", linewidth=2.2, label="Mean")
    ax.fill_between(users, mean - std, mean + std, alpha=0.18, label="Mean ± std")
    ax.plot(users, min_v, marker="v", linewidth=1.8, label="Min")
    ax.plot(users, max_v, marker="^", linewidth=1.8, label="Max")
    ax.set_title("First Audio Latency vs Concurrent Users")
    ax.set_xlabel("Concurrent users")
    ax.set_ylabel("Latency to first audio chunk (ms)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()

    path = out_dir / "latency_vs_users.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_throughput(rows: list[BenchmarkRow], out_dir: Path) -> Path:
    users = np.asarray([row.users for row in rows])
    rps = np.asarray([row.throughput_requests_per_sec for row in rows])
    audio_rtf = np.asarray([row.throughput_audio_sec_per_sec for row in rows])
    max_idx = int(np.argmax(rps))

    fig, ax1 = plt.subplots(figsize=(11, 6.5))
    ax1.plot(users, rps, marker="o", linewidth=2.2, label="Requests/sec")
    ax1.scatter([users[max_idx]], [rps[max_idx]], s=90, zorder=5, label=f"Max: {rps[max_idx]:.2f} req/s")
    ax1.set_title("Throughput vs Concurrent Users")
    ax1.set_xlabel("Concurrent users")
    ax1.set_ylabel("Completed requests/sec")
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(users, audio_rtf, marker="s", linestyle="--", linewidth=1.8, color="tab:green", label="Audio sec/sec")
    ax2.set_ylabel("Generated audio seconds / wall second")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="best")
    fig.tight_layout()

    path = out_dir / "throughput_vs_users.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


async def run(args: argparse.Namespace) -> None:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Missing model: {MODEL_PATH}")
    if not (VOICES_FOLDER / f"{VOICE_ID}.pth").exists():
        raise FileNotFoundError(f"Missing voice: {VOICES_FOLDER / f'{VOICE_ID}.pth'}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    config = StreamingTTSConfig(
        default_batch_size=args.batch_size,
        compile_models=False,
        default_alignment_type=AlignmentType.NONE,
    )
    tts = StreamingTTS(config)
    print(f"Loading {MODEL_ID} from {MODEL_PATH}")
    tts.load_model(MODEL_ID, str(MODEL_PATH), voices_folder=str(VOICES_FOLDER))
    print("Warming up")
    tts.warmup([MODEL_ID], [VOICE_ID], num_warmup_tasks=max(1, min(args.batch_size, 4)))
    async_tts = AsyncStreamingTTS(tts)
    await first_chunk_latency(async_tts)

    rows: list[BenchmarkRow] = []
    try:
        for users in args.users:
            print(f"\n=== {users} concurrent users ===")
            row = await measure_users(async_tts, users, args.repeats)
            rows.append(row)
            print(
                f"latency ms mean={row.latency_mean_ms:.1f} std={row.latency_std_ms:.1f} "
                f"min={row.latency_min_ms:.1f} max={row.latency_max_ms:.1f}; "
                f"throughput={row.throughput_requests_per_sec:.2f} req/s "
                f"({row.throughput_audio_sec_per_sec:.2f} audio sec/s)"
            )
    finally:
        async_tts.stop()
        tts.stop()

    csv_path, json_path = write_results(rows, OUT_DIR)
    latency_plot = plot_latency(rows, OUT_DIR)
    throughput_plot = plot_throughput(rows, OUT_DIR)
    max_row = max(rows, key=lambda row: row.throughput_requests_per_sec)

    print("\nSUMMARY")
    print(f"CSV: {csv_path}")
    print(f"JSON: {json_path}")
    print(f"Latency plot: {latency_plot}")
    print(f"Throughput plot: {throughput_plot}")
    print(
        f"Maximum throughput: {max_row.throughput_requests_per_sec:.2f} requests/sec "
        f"at {max_row.users} concurrent users"
    )
    print(
        f"Maximum audio generation rate at that point: "
        f"{max_row.throughput_audio_sec_per_sec:.2f} audio seconds/sec"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark Tinfer latency and throughput, then save plots.")
    parser.add_argument("--users", type=int, nargs="+", default=[1, 2, 4, 8, 12, 16])
    parser.add_argument("--repeats", type=int, default=2, help="Latency repeats per concurrency level.")
    parser.add_argument("--batch-size", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    asyncio.run(run(parse_args()))


if __name__ == "__main__":
    main()
