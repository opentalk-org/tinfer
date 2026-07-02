from __future__ import annotations

from pathlib import Path
from time import monotonic

import numpy as np

from tinfer.config.engine_config import StreamingTTSConfig
from tinfer.core.engine import StreamingTTS
from tinfer.core.request import AlignmentType, TTSRequest
from tinfer.utils.text_chunker import TextChunker


BASE_DIR = Path("/workspace")
MODEL_ID = "styletts2"
VOICE_ID = "magda_001"
MODEL_PATH = BASE_DIR / "converted_models" / "magda" / "model.pth"
VOICES_FOLDER = BASE_DIR / "converted_models" / "magda" / "voices"


def make_long_text(sentences: int = 80) -> str:
    base = [
        "To jest długi test harmonogramowania syntezy mowy.",
        "Każde kolejne zdanie powinno zostać przetworzone w poprawnej kolejności.",
        "Tekst celowo przekracza domyślną listę długości fragmentów.",
        "Sprawdzamy, czy po wielu fragmentach nie pojawi się błąd indeksu.",
    ]
    return " ".join(base[i % len(base)] for i in range(sentences))


def run_chunker_probe() -> None:
    request = TTSRequest(
        request_id="probe",
        model_id=MODEL_ID,
        voice_id=VOICE_ID,
        chunk_length_schedule=[80, 160, 250, 290],
        min_chunk_length_schedule=[50, 80, 120, 150],
        timeout_trigger_ms=0,
        alignment_type=AlignmentType.NONE,
    )
    request.append_text(make_long_text(60))
    chunker = TextChunker()

    total_committed = 0
    lengths: list[int] = []
    for step in range(12):
        chunks = chunker.get_chunks(request, single_chunk=True)
        if not chunks:
            break
        text_chunk, chunk_index, _is_final, text_span = chunks[0]
        lengths.append(len(text_chunk))
        request.commit_text(len(text_chunk))
        total_committed += len(text_chunk)
        print(
            f"probe step={step} chunk_index={chunk_index} len={len(text_chunk)} "
            f"span={text_span} next_index={request.chunker_state.get('chunk_index')}"
        )

    print(f"probe committed={total_committed} lengths={lengths}")

    overflow_request = TTSRequest(
        request_id="overflow-probe",
        model_id=MODEL_ID,
        voice_id=VOICE_ID,
        chunk_length_schedule=[80, 160, 250, 290],
        min_chunk_length_schedule=[50, 80, 120, 150],
        timeout_trigger_ms=60_000,
        alignment_type=AlignmentType.NONE,
    )
    overflow_request.chunker_state["chunk_index"] = 12
    overflow_request.append_text("A" * 400)
    chunks = chunker.get_chunks(overflow_request, single_chunk=True)
    print(f"overflow probe chunks={len(chunks)} chunk_index={chunks[0][1] if chunks else None}")
    if not chunks:
        raise RuntimeError("Overflow schedule probe did not produce a chunk")


def run_end_to_end() -> None:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Missing model: {MODEL_PATH}")
    if not (VOICES_FOLDER / f"{VOICE_ID}.pth").exists():
        raise FileNotFoundError(f"Missing voice: {VOICES_FOLDER / f'{VOICE_ID}.pth'}")

    text = make_long_text(80)
    config = StreamingTTSConfig(
        compile_models=False,
        default_alignment_type=AlignmentType.NONE,
        default_batch_size=4,
    )
    tts = StreamingTTS(config)
    try:
        print(f"loading model={MODEL_PATH}")
        tts.load_model(MODEL_ID, str(MODEL_PATH), voices_folder=str(VOICES_FOLDER))
        stream = tts.create_stream(MODEL_ID, VOICE_ID, {"alignment_type": AlignmentType.NONE})
        start = monotonic()
        stream.add_text(text)
        stream.force_generate()
        chunks = stream.get_audio()
        elapsed = monotonic() - start
        stream.close()

        errors = [chunk.error for chunk in chunks if getattr(chunk, "error", None)]
        sample_rates = sorted({chunk.sample_rate for chunk in chunks})
        total_samples = int(sum(len(chunk.audio) for chunk in chunks))
        spans = [chunk.text_span for chunk in chunks]
        monotonic_spans = all(a[1] <= b[0] for a, b in zip(spans, spans[1:]))
        first_bad_span_pair = None
        for a, b in zip(spans, spans[1:]):
            if a[1] > b[0]:
                first_bad_span_pair = (a, b)
                break
        covered_chars = sum(max(0, end - start) for start, end in spans)
        duration_s = total_samples / sample_rates[0] if sample_rates and sample_rates[0] else 0.0

        print("end_to_end_summary")
        print(f"chunks={len(chunks)}")
        print(f"errors={errors}")
        print(f"sample_rates={sample_rates}")
        print(f"covered_chars={covered_chars} input_chars={len(text)}")
        print(f"monotonic_spans={monotonic_spans}")
        print(f"first_bad_span_pair={first_bad_span_pair}")
        print(f"total_samples={total_samples}")
        print(f"audio_duration_s={duration_s:.3f}")
        print(f"elapsed_s={elapsed:.3f}")
        print(f"first_spans={spans[:5]}")
        print(f"last_spans={spans[-5:]}")

        if errors:
            raise RuntimeError(f"Inference errors: {errors}")
        if not chunks:
            raise RuntimeError("No chunks returned")
        if not monotonic_spans:
            raise RuntimeError("Chunk text spans are not monotonic")
        if covered_chars <= len(text) * 0.7:
            raise RuntimeError("Unexpectedly low text coverage")
        if not np.isfinite(total_samples) or total_samples <= 0:
            raise RuntimeError("Invalid audio sample count")
    finally:
        tts.stop()


def main() -> None:
    print("=== chunker probe ===")
    run_chunker_probe()
    print("\n=== end-to-end long text ===")
    run_end_to_end()


if __name__ == "__main__":
    main()
