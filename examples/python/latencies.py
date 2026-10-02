import argparse
from concurrent.futures import ThreadPoolExecutor
from time import perf_counter
from urllib.request import urlopen

from config import model_id, voice_id
from utils import VoiceSettings, speech_request


def measure_latency(index: int) -> float:
    request = speech_request("To jest pomiar opóźnienia syntezy mowy.", model_id, voice_id, "/stream", VoiceSettings())
    start = perf_counter()
    with urlopen(request, timeout=180) as response:
        assert response.read(2), "expected audio"
        latency = perf_counter() - start
        response.read()
    print(f"Request {index}: first audio after {latency:.3f}s")
    return latency


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure concurrent HTTP streaming latency")
    parser.add_argument("--requests", type=int, default=3)
    args = parser.parse_args()
    assert args.requests > 0
    with ThreadPoolExecutor(max_workers=args.requests) as workers:
        latencies = list(workers.map(measure_latency, range(args.requests)))
    print(f"Mean latency: {sum(latencies) / len(latencies):.3f}s")


if __name__ == "__main__":
    main()
