from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import librosa
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf

from tmp_tests.alignment_stress_test import check_alignment
from tinfer.config.engine_config import StreamingTTSConfig
from tinfer.core.engine import StreamingTTS
from tinfer.core.request import Alignment, AlignmentType
from tinfer.models.impl.styletts2.alignment.converter import AlignmentConverter
from tinfer.models.impl.styletts2.model.phonemizer import StyleTTS2Phonemizer


matplotlib.use("Agg")

MODEL_ID = "styletts2"
VOICE_ID = "magda_001"
MODEL_PATH = "/workspace/converted_models/magda/model.pth"
VOICES_FOLDER = "/workspace/converted_models/magda/voices"
OUT_DIR = Path("/workspace/tinfer/alignment_artifacts")

CASES = [
    ("dates_times", "Spotkanie: 01.02.2026 r., godz. 08:05-09:30; wersja v2.10.3-beta."),
    ("currency", "Faktura nr FV/2026/07/001: 1 234,56 zł + 23% VAT = 1 518,51 zł; rabat −7,5%."),
    ("url_email", "Dr inż. Kowalski pisał do test@example.com; patrz https://example.org/a-b?x=1#sekcja."),
    ("quotes", "„Cześć” — powiedziała: «To działa… prawda?»"),
    ("math_units", "Wzór: α+β=γ; 3×4=12; 10 µm, 5 m/s, 2 kg/m³, 45° i pH=7,4."),
    ("mixed_names", "Łódź, Żyrardów, São Paulo, München, OpenAI GPT-5 i numer ABC-123/XY."),
    ("whitespace", "Start:\n\t- punkt 1: 10,5 kg;\n\t- punkt 2: 3×4 cm;\n\nKoniec?  Tak...   chyba."),
    ("punctuation_dense", "Halo?!... Czy to naprawdę działa; czy tylko wygląda dobrze: tak, nie, może?"),
    ("long_complex", "Raport 2026-07-01: saldo 1 234,56 zł; koszt 99,90 zł. Mail: finanse+test@example.com. Cytat: „zapłacone?” — tak."),
    ("symbols_fractions", "Pomiary: ½ litra, 3¾ kg, H₂O, CO₂, temperatura −12°C oraz ścieżka C:/tmp/test-01.wav."),
]


def _sanitize_audio(audio: np.ndarray) -> np.ndarray:
    return np.clip(np.nan_to_num(audio.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0), -1.0, 1.0)


def _alignment_from_items(items: list[Any], alignment_type: AlignmentType) -> Alignment:
    alignment = Alignment()
    alignment.items = items
    alignment.type_ = alignment_type
    return alignment


def _draw_combined_alignment_axis(
    ax: Any,
    phoneme_items: list[Any],
    char_items: list[Any],
    mapped_items: list[dict[str, Any]],
    duration_seconds: float,
) -> None:
    ax.set_title("Phoneme and Character Alignments", fontsize=18, fontweight="bold")

    row_pitch = 1.75
    bar_height = 0.46

    def display_label(value: str) -> str:
        value = value.replace("\n", "\\n").replace("\t", "\\t")
        if value == " ":
            return "space"
        return value or " "

    def draw_bar(item: Any, y_pos: float, color: str, label: str | None = None) -> None:
        start_s = item.start_ms / 1000.0
        end_s = item.end_ms / 1000.0
        width = max(0.001, end_s - start_s)
        item_text = getattr(item, "item", "")
        item_color = "tab:red" if item_text.isspace() else color
        ax.barh(
            y_pos,
            width,
            left=start_s,
            height=bar_height,
            alpha=0.82,
            color=item_color,
            edgecolor="black",
            linewidth=0.55,
            zorder=3,
        )
        ax.text(
            start_s + width / 2,
            y_pos,
            display_label(label if label is not None else item_text),
            ha="center",
            va="center",
            fontsize=14,
            fontweight="bold",
            zorder=4,
        )

    if phoneme_items or char_items:
        rows: list[tuple[str, list[Any], int, int]] = []
        for mapped_idx, mapped in enumerate(mapped_items):
            original_start = int(mapped.get("original_start", 0))
            original_end = int(mapped.get("original_end", original_start))
            char_group = [
                item
                for item in char_items
                if item.char_start >= original_start and item.char_end <= original_end
            ]
            if not char_group:
                continue

            row_start_ms = char_group[0].start_ms
            row_end_ms = char_group[-1].end_ms
            phoneme_group = [
                item
                for item in phoneme_items
                if item.start_ms >= row_start_ms and item.start_ms < row_end_ms
            ]
            row_text = str(mapped.get("original_text", ""))
            if not row_text:
                row_text = "".join(item.item for item in char_group)
            rows.append((row_text, phoneme_group, original_start, original_end))

        for row_idx, (row_text, phoneme_group, original_start, original_end) in enumerate(rows):
            y_base = (len(rows) - row_idx - 1) * row_pitch
            char_y = y_base + 0.36
            phoneme_y = y_base - 0.36
            if row_idx % 2 == 0:
                ax.axhspan(y_base - 0.86, y_base + 0.86, color="0.95", zorder=0)

            char_group = [
                item
                for item in char_items
                if item.char_start >= original_start and item.char_end <= original_end
            ]
            for item in char_group:
                draw_bar(item, char_y, "tab:orange")

            for item in phoneme_group:
                draw_bar(item, phoneme_y, "tab:blue")

        max_y = (len(rows) - 1) * row_pitch if rows else 0
        ax.set_ylim(-1.05, max(1.05, max_y + 1.05))
        ax.set_yticks([idx * row_pitch for idx in range(len(rows))])
        ax.set_yticklabels(
            [display_label(row[0]) for row in reversed(rows)],
            fontsize=17,
            fontweight="bold",
        )
    else:
        ax.text(
            0.5,
            0.5,
            "No alignments available",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=16,
        )

    ax.set_xlim(0, duration_seconds)
    ax.tick_params(axis="both", labelsize=14)
    ax.grid(axis="x", color="0.78", linestyle="--", linewidth=0.8)
    ax.set_xlabel("Time (seconds)", fontsize=16)
    ax.set_ylabel("Text span", fontsize=16)


def plot_alignment(
    audio: np.ndarray,
    sample_rate: int,
    phoneme_alignment: Alignment | None,
    char_alignment: Alignment | None,
    mapped_items: list[dict[str, Any]],
    title: str,
    output_path: Path,
) -> None:
    duration_seconds = len(audio) / sample_rate
    row_count = max(1, len(mapped_items))
    figure_height = min(90, max(22, 9 + row_count * 1.05))
    fig, axes = plt.subplots(
        2,
        1,
        figsize=(70, figure_height),
        gridspec_kw={"height_ratios": [2, max(10, row_count * 1.05)]},
    )
    fig.suptitle(title, fontsize=22, fontweight="bold")

    spectrogram = librosa.stft(audio.astype(np.float32), n_fft=2048, hop_length=512)
    magnitude_db = librosa.amplitude_to_db(np.abs(spectrogram), ref=np.max)
    axes[0].imshow(
        magnitude_db,
        aspect="auto",
        origin="lower",
        extent=[0, duration_seconds, 0, sample_rate / 2],
        cmap="viridis",
    )
    axes[0].set_title("Speech Spectrogram", fontsize=18, fontweight="bold")
    axes[0].set_ylabel("Frequency (Hz)", fontsize=16)
    axes[0].set_xlim(0, duration_seconds)
    axes[0].tick_params(axis="both", labelsize=14)

    _draw_combined_alignment_axis(
        axes[1],
        list(phoneme_alignment.items if phoneme_alignment else []),
        list(char_alignment.items if char_alignment else []),
        mapped_items,
        duration_seconds,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


async def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tts = StreamingTTS(StreamingTTSConfig(compile_models=False))
    tts.load_model(MODEL_ID, MODEL_PATH, voices_folder=VOICES_FOLDER)
    phonemizer = StyleTTS2Phonemizer(language="pl")

    report = []
    try:
        for index, (case_id, text) in enumerate(CASES, start=1):
            case_dir = OUT_DIR / f"{index:02d}_{case_id}"
            case_dir.mkdir(parents=True, exist_ok=True)
            print(f"Generating {index:02d}_{case_id}")

            chunk = await tts.generate_full(MODEL_ID, VOICE_ID, text, {"alignment_type": AlignmentType.PHONEME})
            audio = _sanitize_audio(np.asarray(chunk.audio))
            mapped = phonemizer.align_text_with_original_spans(text)
            char_items = AlignmentConverter.phoneme_to_char_mapped(
                list(chunk.alignments.items if chunk.alignments else []),
                text,
                mapped,
            )
            char_alignment = _alignment_from_items(char_items, AlignmentType.CHAR)
            wav_path = case_dir / f"{case_id}.wav"
            plot_path = case_dir / f"{case_id}_alignment.png"
            sf.write(wav_path, audio, chunk.sample_rate)
            plot_alignment(audio, chunk.sample_rate, chunk.alignments, char_alignment, mapped, case_id, plot_path)

            audio_duration_ms = len(audio) / chunk.sample_rate * 1000
            checked = check_alignment(text, audio_duration_ms, char_alignment, AlignmentType.CHAR)
            report.append(
                {
                    "id": case_id,
                    "text": text,
                    "wav_path": str(wav_path),
                    "plot_path": str(plot_path),
                    "issues": checked["issues"],
                    "item_count": checked["item_count"],
                    "audio_duration_ms": checked["audio_duration_ms"],
                    "last_end_ms": checked["last_end_ms"],
                }
            )
    finally:
        tts.stop()

    report_path = OUT_DIR / "alignment_artifacts_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"cases": len(report), "cases_with_issues": sum(bool(r["issues"]) for r in report), "report_path": str(report_path)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
