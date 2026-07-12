from dataclasses import dataclass
import json
import random
from urllib.request import urlopen


DATASET_ID = "b3ed6328-88b0-4b7f-bb4e-c37319bdbe5f"
BACKEND_URL = "http://127.0.0.1:8001"


@dataclass(frozen=True)
class BackendSegment:
    segment_id: str
    phonemes: str
    duration_seconds: float


@dataclass(frozen=True)
class DurationSample:
    sample_id: str
    tokens: list[int]
    duration_seconds: float


@dataclass(frozen=True)
class ContextPair:
    short_tokens: list[int]
    long_tokens: list[int]
    shared_length: int


@dataclass(frozen=True)
class SyntheticSample:
    tokens: list[int]
    duration_seconds: float


def fetch_backend_segments() -> list[BackendSegment]:
    endpoint = (
        f"{BACKEND_URL}/audio-files?dataset={DATASET_ID}"
        "&limit=200&sort=name"
    )
    with urlopen(endpoint) as response:
        rows = json.load(response)["rows"]
    segments = []
    for row in rows:
        with urlopen(f"{BACKEND_URL}/audio-files/{row['id']}") as response:
            detail = json.load(response)
        for segment in detail["segment_preview"]:
            segments.append(
                BackendSegment(
                    segment["id"],
                    segment["phon"],
                    float(segment["end"] - segment["start"]),
                )
            )
    return segments


def tokenize_segments(segments, phonemizer) -> list[DurationSample]:
    samples = []
    for segment in segments:
        tokens = [0, *phonemizer.tokenize(segment.phonemes)]
        samples.append(
            DurationSample(
                segment.segment_id,
                tokens,
                segment.duration_seconds,
            )
        )
    return samples


def build_context_pair(
    tokens: list[int],
    short_length: int,
    terminal_token: int,
) -> ContextPair:
    assert 3 < short_length < len(tokens)
    short_tokens = [*tokens[: short_length - 1], terminal_token]
    return ContextPair(short_tokens, tokens, short_length - 1)


def concatenate_samples(
    samples: list[DurationSample],
    maximum_tokens: int,
    terminal_token: int,
    randomizer: random.Random,
) -> SyntheticSample:
    ordered = randomizer.sample(samples, len(samples))
    tokens = [0]
    duration = 0.0
    for sample in ordered:
        content = sample.tokens[1:-1]
        available = maximum_tokens - len(tokens) - 1
        included = content[:available]
        tokens.extend(included)
        tokens.append(terminal_token)
        duration += sample.duration_seconds * len(included) / len(content)
        if len(tokens) == maximum_tokens:
            break
    return SyntheticSample(tokens, duration)
