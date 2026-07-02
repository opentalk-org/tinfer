from __future__ import annotations

import json
import asyncio
from pathlib import Path
from typing import Any

from tmp_tests.alignment_stress_test import OUT_DIR, run_case
from tinfer.config.engine_config import StreamingTTSConfig
from tinfer.core.engine import StreamingTTS


EXTENDED_CASES = [
    {
        "id": "dates_times_versions",
        "text": "Spotkanie: 01.02.2026 r., godz. 08:05-09:30; wersja v2.10.3-beta.",
    },
    {
        "id": "currency_percentages",
        "text": "Faktura nr FV/2026/07/001: 1 234,56 zł + 23% VAT = 1 518,51 zł; rabat −7,5%.",
    },
    {
        "id": "long_multi_sentence_complex",
        "text": (
            "Pierwszy akapit ma datę 31.12.2025, kwotę 99,90 zł i adres test+alert@example.co.uk. "
            "Drugi akapit pyta: „Czy https://example.org/a-b?x=1#sekcja działa?” — tak, działa... chyba! "
            "Trzeci fragment zawiera Łódź, São Paulo, München, H₂O, CO₂, ½ litra oraz temperaturę −12°C."
        ),
    },
    {
        "id": "punctuation_heavy_nested",
        "text": "Powiedział: „To jest «bardzo» ważne?!... Naprawdę; serio: tak”.",
    },
    {
        "id": "paths_ids_codes",
        "text": "Plik C:/tmp/test-01.wav, ID ABC-123/XY, hash a1:b2:c3, tel. +48 123 456 789.",
    },
    {
        "id": "math_units_symbols",
        "text": "Wzór: α+β=γ; 3×4=12; 10 µm, 5 m/s, 2 kg/m³, 45° i pH=7,4.",
    },
    {
        "id": "repeated_chunk_boundaries",
        "text": " ".join(
            [
                "Sekcja 1: koszt 12,50 zł; termin 2026-07-01.",
                "Sekcja 2: mail a.b+c@example.com i link https://a-b.pl/x.",
                "Sekcja 3: „cytat” — pauza, wielokropek… oraz pytanie?",
                "Sekcja 4: Łódź, Żyrardów, São Paulo, München.",
                "Sekcja 5: ½ + ¾ = 1¼; temperatura −5°C.",
            ]
        ),
    },
    {
        "id": "whitespace_complex_long",
        "text": "Start:\n\t- punkt 1: 10,5 kg;\n\t- punkt 2: 3×4 cm;\n\nKoniec?  Tak...   chyba.",
    },
    {
        "id": "long_supported_finance_dates",
        "text": "Raport 2026-07-01: saldo 1 234,56 zł; koszt 99,90 zł; VAT 23%. Mail: finanse+test@example.com. Cytat: „zapłacone?” — tak.",
    },
    {
        "id": "long_supported_punctuation",
        "text": "Lista: A/B, C-D, E_F; pytanie?!... Odpowiedź: «może», „raczej tak”, albo nie; godz. 08:05.",
    },
    {
        "id": "long_supported_units_symbols",
        "text": "Pomiary: ½ litra, 3¾ kg, 10 µm, 5 m/s, −12°C, 45°, pH=7,4; lokalizacje: Łódź, München, São Paulo.",
    },
]


async def async_main() -> None:
    extended_dir = OUT_DIR / "extended"
    extended_dir.mkdir(parents=True, exist_ok=True)

    tts = StreamingTTS(StreamingTTSConfig(compile_models=False))
    tts.load_model(
        "styletts2",
        "/workspace/converted_models/magda/model.pth",
        voices_folder="/workspace/converted_models/magda/voices",
    )

    results: list[dict[str, Any]] = []
    try:
        for case in EXTENDED_CASES:
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

    report_path = extended_dir / "alignment_extended_stress_report.json"
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
