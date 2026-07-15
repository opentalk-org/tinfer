import argparse
import asyncio
import json
import statistics
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter

import grpc

REPOSITORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY / "tinfer"))

from tinfer.server.grpc import styletts_pb2, styletts_pb2_grpc

SAMPLE_RATE = 24_000
BENCHMARK_TEXT = (
    "Ten test pełnego potoku syntezy mowy. Mierzymy opóźnienie pierwszego "
    "fragmentu audio oraz szybkość generowania całej wypowiedzi."
)


@dataclass(frozen=True)
class RequestMeasurement:
    ttfb_ms: float
    total_seconds: float
    audio_seconds: float

    @property
    def rtf(self) -> float:
        return self.total_seconds / self.audio_seconds


@dataclass(frozen=True)
class RunMeasurement:
    wall_seconds: float
    requests: list[RequestMeasurement]

    @property
    def batch_wall_rtf(self) -> float:
        return self.wall_seconds / max(item.audio_seconds for item in self.requests)

    @property
    def throughput_rtf(self) -> float:
        return self.wall_seconds / sum(item.audio_seconds for item in self.requests)


@dataclass(frozen=True)
class BatchResult:
    batch_size: int
    ttfb_mean_ms: float
    ttfb_p95_ms: float
    request_rtf_mean: float
    batch_wall_rtf_mean: float
    throughput_rtf_mean: float
    audio_seconds_mean: float


async def measure_request(
    stub: styletts_pb2_grpc.StyleTTSServiceStub,
    request: styletts_pb2.SynthesizeRequest,
    release: asyncio.Event,
) -> RequestMeasurement:
    await release.wait()
    started = perf_counter()
    first_byte = None
    audio_bytes = 0
    async for response in stub.SynthesizeStream(request):
        if response.audio_data and first_byte is None:
            first_byte = perf_counter()
        audio_bytes += len(response.audio_data)
    finished = perf_counter()
    if first_byte is None or audio_bytes == 0:
        raise RuntimeError("gRPC stream returned no audio")
    return RequestMeasurement(
        ttfb_ms=(first_byte - started) * 1_000,
        total_seconds=finished - started,
        audio_seconds=audio_bytes / 2 / SAMPLE_RATE,
    )


async def measure_run(
    stub: styletts_pb2_grpc.StyleTTSServiceStub,
    request: styletts_pb2.SynthesizeRequest,
    batch_size: int,
) -> RunMeasurement:
    release = asyncio.Event()
    tasks = [asyncio.create_task(measure_request(stub, request, release)) for _ in range(batch_size)]
    await asyncio.sleep(0)
    started = perf_counter()
    release.set()
    requests = await asyncio.gather(*tasks)
    return RunMeasurement(wall_seconds=perf_counter() - started, requests=requests)


def percentile_95(values: list[float]) -> float:
    ordered = sorted(values)
    return ordered[max(0, (95 * len(ordered) + 99) // 100 - 1)]


def summarize(batch_size: int, runs: list[RunMeasurement]) -> BatchResult:
    requests = [request for run in runs for request in run.requests]
    return BatchResult(
        batch_size=batch_size,
        ttfb_mean_ms=statistics.fmean(item.ttfb_ms for item in requests),
        ttfb_p95_ms=percentile_95([item.ttfb_ms for item in requests]),
        request_rtf_mean=statistics.fmean(item.rtf for item in requests),
        batch_wall_rtf_mean=statistics.fmean(run.batch_wall_rtf for run in runs),
        throughput_rtf_mean=statistics.fmean(run.throughput_rtf for run in runs),
        audio_seconds_mean=statistics.fmean(item.audio_seconds for item in requests),
    )


def table(results: list[BatchResult]) -> str:
    lines = [
        "| Batch | TTFB mean | TTFB p95 | Request RTF | Batch wall RTF | Throughput RTF | Audio |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for result in results:
        lines.append(
            f"| {result.batch_size} | {result.ttfb_mean_ms:.1f} ms | "
            f"{result.ttfb_p95_ms:.1f} ms | {result.request_rtf_mean:.3f} | "
            f"{result.batch_wall_rtf_mean:.3f} | {result.throughput_rtf_mean:.3f} | "
            f"{result.audio_seconds_mean:.2f} s |"
        )
    return "\n".join(lines)


async def benchmark(arguments: argparse.Namespace) -> list[BatchResult]:
    request = styletts_pb2.SynthesizeRequest(
        text=BENCHMARK_TEXT,
        config=styletts_pb2.SynthesisConfig(
            model_id=arguments.model,
            voice_id=arguments.voice,
            sample_rate_hz=SAMPLE_RATE,
            language=arguments.language,
        ),
    )
    async with grpc.aio.insecure_channel(arguments.address) as channel:
        await channel.channel_ready()
        stub = styletts_pb2_grpc.StyleTTSServiceStub(channel)
        for _ in range(arguments.warmup):
            await measure_run(stub, request, 1)
        results = []
        for batch_size in arguments.batch_sizes:
            runs = [await measure_run(stub, request, batch_size) for _ in range(arguments.runs)]
            result = summarize(batch_size, runs)
            results.append(result)
            print(table([result]), flush=True)
        return results


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark the complete tinfer gRPC streaming path")
    parser.add_argument("--address", default="127.0.0.1:50051")
    parser.add_argument("--model", default="olam")
    parser.add_argument("--voice", default="ola")
    parser.add_argument("--language", default="pl")
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 2, 4, 8, 16])
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    options = arguments()
    results = asyncio.run(benchmark(options))
    print(table(results))
    if options.output is not None:
        options.output.write_text(json.dumps([asdict(result) for result in results], indent=2) + "\n")


if __name__ == "__main__":
    main()
