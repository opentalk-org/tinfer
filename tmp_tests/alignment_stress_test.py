from __future__ import annotations

import json
import math
import asyncio
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

from tinfer.config.engine_config import StreamingTTSConfig
from tinfer.core.engine import StreamingTTS
from tinfer.core.request import AlignmentType


BASE_DIR = Path("/workspace")
MODEL_ID = "styletts2"
VOICE_ID = "magda_001"
MODEL_PATH = BASE_DIR / "converted_models" / "magda" / "model.pth"
VOICES_FOLDER = BASE_DIR / "converted_models" / "magda" / "voices"
OUT_DIR = BASE_DIR / "tinfer" / "alignment_stress_outputs"


CASES = [
    {
        "id": "pl_diacritics_dense",
        "text": "Zażółć gęślą jaźń. Źdźbło, żółw, chrząszcz i szczęście.",
    },
    {
        "id": "quotes_dashes_ellipsis",
        "text": "„Cześć” — powiedziała: «To działa… prawda?»",
    },
    {
        "id": "numbers_units_currency",
        "text": "Mam 12,5 kg jabłek, 3½ litra wody i 99,90 zł.",
    },
    {
        "id": "abbrev_email_url",
        "text": "Dr inż. Kowalski pisał do test@example.com o godz. 8:05; patrz https://example.org/a-b.",
    },
    {
        "id": "emoji_symbols_math",
        "text": "Test 😀: alfa+beta=γ, 2×2=4, temperatura −5°C.",
    },
    {
        "id": "mixed_language_names",
        "text": "Agnieszka mówi: OpenAI, Łódź, São Paulo, München i Żyrardów.",
    },
    {
        "id": "spacing_newlines_tabs",
        "text": "Pierwsza linia.\nDruga\tlinia z  wieloma   spacjami.",
    },
    {
        "id": "punctuation_clusters",
        "text": "Halo?!... Czy to naprawdę działa; czy tylko wygląda dobrze?",
    },
]


def item_to_dict(item: Any) -> dict[str, Any]:
    return {
        "item": item.item,
        "char_start": item.char_start,
        "char_end": item.char_end,
        "start_ms": item.start_ms,
        "end_ms": item.end_ms,
    }


def check_alignment(text: str, audio_duration_ms: float, alignment: Any, alignment_type: AlignmentType) -> dict[str, Any]:
    items = list(alignment.items if alignment else [])
    issues: list[str] = []

    if not items:
        issues.append("empty_alignment")
        return {
            "type": alignment_type.value,
            "item_count": 0,
            "issues": issues,
            "joined_items": "",
            "first_items": [],
            "last_items": [],
        }

    previous_start = -math.inf
    previous_end = -math.inf
    for idx, item in enumerate(items):
        if item.start_ms < 0 or item.end_ms < 0:
            issues.append(f"negative_time_at_{idx}")
        if item.end_ms < item.start_ms:
            issues.append(f"negative_duration_at_{idx}:{item.item!r}")
        if item.start_ms < previous_start:
            issues.append(f"start_time_regressed_at_{idx}:{item.item!r}")
        if item.start_ms < previous_end - 1:
            issues.append(f"overlap_at_{idx}:{item.item!r}")
        previous_start = item.start_ms
        previous_end = item.end_ms

    last_end_ms = items[-1].end_ms
    if last_end_ms > audio_duration_ms + 120:
        issues.append(f"alignment_exceeds_audio:last_end={last_end_ms} audio={audio_duration_ms:.1f}")
    if audio_duration_ms - last_end_ms > 1000:
        issues.append(f"large_trailing_audio:last_end={last_end_ms} audio={audio_duration_ms:.1f}")

    joined = "".join(item.item for item in items)
    normalized_joined = " ".join(joined.split())
    normalized_text = " ".join(text.strip().split())

    if alignment_type == AlignmentType.CHAR:
        if joined != text.strip():
            issues.append("char_join_not_exact_original_text")
        if normalized_joined != normalized_text:
            issues.append("char_join_not_normalized_original_text")

        bad_offsets = []
        for idx, item in enumerate(items):
            if not (0 <= item.char_start <= item.char_end <= len(text)):
                bad_offsets.append((idx, item.item, item.char_start, item.char_end, "out_of_range"))
                continue
            source_slice = text[item.char_start : item.char_end]
            if source_slice != item.item:
                bad_offsets.append((idx, item.item, item.char_start, item.char_end, source_slice))
        if bad_offsets:
            issues.append(f"char_offsets_do_not_match_source:{bad_offsets[:8]}")

    if alignment_type == AlignmentType.WORD:
        if normalized_joined != normalized_text:
            issues.append("word_items_do_not_reconstruct_normalized_original_text")
        bad_offsets = []
        for idx, item in enumerate(items):
            if not (0 <= item.char_start <= item.char_end <= len(text)):
                bad_offsets.append((idx, item.item, item.char_start, item.char_end, "out_of_range"))
                continue
            source_slice = text[item.char_start : item.char_end]
            if source_slice != item.item:
                bad_offsets.append((idx, item.item, item.char_start, item.char_end, source_slice))
        if bad_offsets:
            issues.append(f"word_offsets_do_not_match_source:{bad_offsets[:8]}")

    return {
        "type": alignment_type.value,
        "item_count": len(items),
        "issues": issues,
        "audio_duration_ms": round(audio_duration_ms, 3),
        "last_end_ms": last_end_ms,
        "joined_items": joined,
        "normalized_joined": normalized_joined,
        "normalized_text": normalized_text,
        "first_items": [item_to_dict(item) for item in items[:12]],
        "last_items": [item_to_dict(item) for item in items[-8:]],
    }


async def run_case(tts: StreamingTTS, case: dict[str, str]) -> dict[str, Any]:
    case_dir = OUT_DIR / case["id"]
    case_dir.mkdir(parents=True, exist_ok=True)

    result: dict[str, Any] = {
        "id": case["id"],
        "text": case["text"],
        "word": None,
        "char": None,
        "errors": [],
    }

    for alignment_type in (AlignmentType.WORD, AlignmentType.CHAR):
        try:
            chunk = await tts.generate_full(MODEL_ID, VOICE_ID, case["text"], {"alignment_type": alignment_type})
            audio = np.asarray(chunk.audio)
            audio_duration_ms = len(audio) / chunk.sample_rate * 1000
            wav_path = case_dir / f"{alignment_type.value}.wav"
            sf.write(str(wav_path), audio, chunk.sample_rate)
            checked = check_alignment(case["text"], audio_duration_ms, chunk.alignments, alignment_type)
            checked["wav_path"] = str(wav_path)
            result[alignment_type.value] = checked
        except Exception as exc:
            result["errors"].append({"type": alignment_type.value, "error": repr(exc)})

    return result


async def async_main() -> None:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Missing model: {MODEL_PATH}")
    if not (VOICES_FOLDER / f"{VOICE_ID}.pth").exists():
        raise FileNotFoundError(f"Missing voice: {VOICES_FOLDER / f'{VOICE_ID}.pth'}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tts = StreamingTTS(StreamingTTSConfig(compile_models=False))
    tts.load_model(MODEL_ID, str(MODEL_PATH), voices_folder=str(VOICES_FOLDER))

    results = []
    try:
        for case in CASES:
            print(f"\n=== {case['id']} ===")
            result = await run_case(tts, case)
            results.append(result)
            for key in ("word", "char"):
                checked = result.get(key) or {}
                print(
                    f"{key}: items={checked.get('item_count')} "
                    f"last_end={checked.get('last_end_ms')} "
                    f"audio_ms={checked.get('audio_duration_ms')} "
                    f"issues={checked.get('issues')}"
                )
            if result["errors"]:
                print(f"errors={result['errors']}")
    finally:
        tts.stop()

    report_path = OUT_DIR / "alignment_stress_report.json"
    report_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    summary = {
        "cases": len(results),
        "word_cases_with_issues": sum(bool((r.get("word") or {}).get("issues")) for r in results),
        "char_cases_with_issues": sum(bool((r.get("char") or {}).get("issues")) for r in results),
        "errors": sum(len(r.get("errors", [])) for r in results),
        "report_path": str(report_path),
    }
    print("\nSUMMARY")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    asyncio.run(async_main())
