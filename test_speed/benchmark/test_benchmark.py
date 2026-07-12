from pathlib import Path
from types import SimpleNamespace
import logging
import tempfile
import unittest
import zipfile

import numpy as np
import torch.nn as nn

from tinfer.core.request import AlignmentItem
from tinfer.models.impl.styletts2.alignment.alignment import StyleTTS2AlignmentParser
from tinfer.models.impl.styletts2.model.inference_config import StyleTTS2Params
from tinfer.models.impl.styletts2.voice.encoder import StyleTTS2VoiceEncoder

from test_speed.benchmark.benchmark_data import (
    PhonemeMetric,
    ReferenceDuration,
    RequestMetric,
    TextInput,
    select_names,
    summarize_phonemes,
)
from test_speed.benchmark.benchmark_inference import (
    archive_wav_names,
    extract_selected,
    measure_result,
    synthesize_all,
)
from test_speed.benchmark.benchmark_reporting import (
    mean_rates_by_voice,
    shared_histogram_edges,
    write_reports,
)
from test_speed.benchmark.benchmark_style import StyleEmbeddingNorm
from test_speed.benchmark.run_benchmark import (
    PROFILES,
    configure_progress_output,
    copy_profile_inputs,
)


class RecordingModel:
    _max_styletts_tokens = 512

    def __init__(self) -> None:
        self.params: list[dict[str, object]] = []

    def _text_token_count(
        self,
        text: str,
        params: StyleTTS2Params,
    ) -> int:
        return len(text)

    def generate(
        self,
        text: str,
        context: dict[str, object],
        params: dict[str, object],
        metadata: dict[str, object],
    ) -> SimpleNamespace:
        self.params.append(params)
        return SimpleNamespace(
            data=np.zeros(240, dtype=np.float32),
            sample_rate=24000,
            metadata={
                "word_alignments": [AlignmentItem("n", 0, 1, 0, 25)]
            },
        )


class DataTests(unittest.TestCase):
    def test_selection_is_seeded_and_sorted(self) -> None:
        names = [f"v{i}" for i in range(30)]
        selected = select_names(names, 20, 7)

        self.assertEqual(selected, select_names(names, 20, 7))
        self.assertEqual(selected, sorted(selected))
        self.assertEqual(len(selected), 20)

    def test_selection_rejects_insufficient_inputs(self) -> None:
        with self.assertRaisesRegex(ValueError, "Need 20 inputs, found 1"):
            select_names(["only"], 20, 7)

    def test_summary_groups_each_phoneme(self) -> None:
        rows = summarize_phonemes(
            [
                PhonemeMetric("a", 0.025, "v", "t"),
                PhonemeMetric("a", 0.075, "v", "t"),
            ]
        )

        self.assertEqual(rows[0].count, 2)
        self.assertAlmostEqual(rows[0].average_seconds, 0.05)
        self.assertAlmostEqual(rows[0].minimum_seconds, 0.025)
        self.assertAlmostEqual(rows[0].maximum_seconds, 0.075)

class InferenceTests(unittest.TestCase):
    def test_synthesis_forwards_diffusion_flag(self) -> None:
        model = RecordingModel()

        with tempfile.TemporaryDirectory() as directory:
            synthesize_all(
                model,
                ["voice"],
                [TextInput("short", "No", 4)],
                Path(directory),
                False,
                "en-us", False,
            )

        self.assertEqual(
            model.params,
            [{"use_diffusion": False, "language": "en-us"}],
        )

    def test_voice_encoder_preserves_attribute_model_access_after_move(self) -> None:
        encoder = StyleTTS2VoiceEncoder(
            model={
                "style_encoder": nn.Linear(1, 1),
                "predictor_encoder": nn.Linear(1, 1),
            },
            device="cpu",
            sample_rate=24000,
        )

        encoder.to("cpu")

        self.assertIsInstance(encoder.model.style_encoder, nn.Linear)
        self.assertIsInstance(encoder.model.predictor_encoder, nn.Linear)

    def test_phoneme_alignment_excludes_beginning_of_sequence_duration(self) -> None:
        predictor_alignment = np.asarray(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 1.0],
            ]
        )

        alignments = StyleTTS2AlignmentParser().parse_from_pred_aln_trg(
            predictor_alignment,
            [1],
            "a",
            "a",
        )

        self.assertEqual(len(alignments), 1)
        self.assertEqual(alignments[0].item, "a")
        self.assertEqual(alignments[0].end_ms - alignments[0].start_ms, 50)

    def test_measure_result_uses_predictor_alignment(self) -> None:
        metadata = {"word_alignments": [AlignmentItem("a", 0, 1, 0, 25)]}
        result = SimpleNamespace(metadata=metadata)

        request, phonemes = measure_result(
            result,
            "v",
            TextInput("short", "No", 4),
            Path("a.wav"),
        )

        self.assertEqual(request.phoneme_count, 1)
        self.assertAlmostEqual(request.phonemes_per_second, 40.0)
        self.assertAlmostEqual(phonemes[0].duration_seconds, 0.025)

    def test_merged_model_windows_are_rejected(self) -> None:
        result = SimpleNamespace(
            metadata={"window_count": 2, "word_alignments": []}
        )

        with self.assertRaisesRegex(RuntimeError, "chunked"):
            measure_result(
                result,
                "v",
                TextInput("short", "No", 4),
                Path("a.wav"),
            )

    def test_archive_filters_metadata_and_extracts_selected_wavs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = root / "voices.zip"
            with zipfile.ZipFile(archive, "w") as bundle:
                bundle.writestr("voice.wav", b"RIFFdata")
                bundle.writestr("__MACOSX/._voice.wav", b"metadata")
                bundle.writestr("notes.txt", b"ignore")

            self.assertEqual(archive_wav_names(archive), ["voice.wav"])
            extracted = extract_selected(archive, ["voice.wav"], root / "out")

            self.assertEqual(extracted, [root / "out/voice.wav"])
            self.assertEqual(extracted[0].read_bytes(), b"RIFFdata")


class ReportingTests(unittest.TestCase):
    def test_voice_histogram_values_average_each_voice(self) -> None:
        requests = [
            RequestMetric("a", "x", "x", 1, 7, 1, 0.05, 20.0, "a.wav"),
            RequestMetric("a", "y", "y", 1, 9, 1, 0.025, 40.0, "b.wav"),
            RequestMetric("b", "x", "x", 1, 7, 1, 0.025, 40.0, "c.wav"),
        ]

        self.assertEqual(mean_rates_by_voice(requests), [30.0, 40.0])

    def test_shared_histogram_edges_cover_both_profiles(self) -> None:
        edges = shared_histogram_edges(
            [10.2, 11.8],
            [9.1, 12.3],
            bin_width=0.25,
        )

        np.testing.assert_array_equal(
            edges,
            np.arange(9.0, 12.75, 0.25),
        )

    def test_progress_output_suppresses_model_debug_logs(self) -> None:
        configure_progress_output()

        self.assertEqual(logging.getLogger().level, logging.WARNING)

    def test_report_writes_global_and_voice_artifacts(self) -> None:
        requests = [
            RequestMetric("v", "short", "No", 2, 4, 1, 0.025, 40.0, "a.wav")
        ]
        phonemes = [PhonemeMetric("a", 0.025, "v", "short")]

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            index_path = write_reports(
                root,
                requests,
                phonemes,
                [ReferenceDuration("v", 1.0)],
                [StyleEmbeddingNorm("v", 1.0, 2.0, 3.0)],
                ["v"],
                np.asarray([39.0, 40.0, 41.0]),
            )

            self.assertTrue(
                (root / "summary/global_phoneme_durations.csv").is_file()
            )
            self.assertTrue(
                (root / "summary/global_phonemes_per_second.png").is_file()
            )
            self.assertTrue((root / "summary/v_phoneme_durations.csv").is_file())
            self.assertTrue(
                (root / "summary/phonemes_per_second_by_voice.png").is_file()
            )
            self.assertTrue(
                (root / "summary/phonemes_per_second_all_runs.png").is_file()
            )
            self.assertTrue(
                (
                    root
                    / "summary/reference_duration_vs_phonemes_per_second_all_runs.png"
                ).is_file()
            )
            self.assertTrue(
                (
                    root
                    / "summary/reference_duration_vs_mean_phonemes_per_second_by_voice.png"
                ).is_file()
            )
            self.assertEqual(index_path, root / "summary/README.md")


class RunnerTests(unittest.TestCase):
    def test_profiles_enable_and_disable_diffusion(self) -> None:
        self.assertEqual(
            [(profile.name, profile.use_diffusion) for profile in PROFILES],
            [("diffusion", True), ("no_diffusion", False)],
        )

    def test_profile_inputs_are_copied_into_second_folder(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            destination = root / "destination"
            (source / "references").mkdir(parents=True)
            (source / "embeddings").mkdir()
            (source / "references/voice.wav").write_bytes(b"wave")
            (source / "embeddings/voice.pth").write_bytes(b"style")

            copy_profile_inputs(source, destination)

            self.assertEqual(
                (destination / "references/voice.wav").read_bytes(),
                b"wave",
            )
            self.assertEqual(
                (destination / "embeddings/voice.pth").read_bytes(),
                b"style",
            )


if __name__ == "__main__":
    unittest.main()
