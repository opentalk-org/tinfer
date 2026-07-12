from importlib import import_module, util
from inspect import signature
from pathlib import Path
from types import ModuleType
import json
import tempfile
import unittest

import numpy as np
import torch

from test_speed.benchmark.benchmark_data import PhonemeMetric, ReferenceDuration, RequestMetric
from test_speed.benchmark.benchmark_reporting import write_reports


class RecordingVoiceModel:
    def __init__(self, vectors: dict[str, torch.Tensor] | None = None) -> None:
        self.loaded: dict[str, torch.Tensor] = {}
        self.vectors = vectors or {}

    def load_voice_from_vector(
        self,
        voice_id: str,
        vector: torch.Tensor,
    ) -> None:
        self.loaded[voice_id] = vector

    def get_voice(self, voice_id: str) -> torch.Tensor:
        return self.vectors[voice_id]


class OptionalFeatureTest(unittest.TestCase):
    def require_module(self, name: str) -> ModuleType:
        self.assertIsNotNone(util.find_spec(name), f"missing module {name}")
        return import_module(name)


class AgnieszkaVoiceTests(OptionalFeatureTest):
    def test_prepared_vectors_use_json_reference_durations(self) -> None:
        module = self.require_module("test_speed.benchmark.benchmark_speakers")
        source_type = module.VectorVoiceSource
        prepare = module.prepare_vector_voices

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            voices_dir = root / "voices"
            voices_dir.mkdir()
            metadata_path = root / "metadata.json"
            output_dir = root / "embeddings"
            torch.save({"tensor": torch.ones(256)}, voices_dir / "a.pth")
            torch.save({"tensor": torch.full((256,), 2.0)}, voices_dir / "b.pth")
            torch.save({"tensor": torch.zeros(256)}, voices_dir / "other.pth")
            metadata_path.write_text(
                json.dumps(
                    {
                        "agnieszka-best": [
                            {"file": "folder/a.wav", "duration": 3.25},
                            {"file": "folder/b.wav", "duration": 6.5},
                        ]
                    }
                )
            )
            model = RecordingVoiceModel()

            prepared = prepare(
                model,
                source_type(voices_dir, metadata_path, "agnieszka-best"),
                output_dir,
                count=2,
                seed=0,
            )

            self.assertEqual(prepared.voice_ids, ["a", "b"])
            self.assertEqual(
                [item.duration_seconds for item in prepared.reference_durations],
                [3.25, 6.5],
            )
            self.assertEqual(set(model.loaded), {"a", "b"})
            self.assertEqual(
                sorted(path.name for path in output_dir.glob("*.pth")),
                ["a.pth", "b.pth"],
            )

    def test_single_vector_voice_needs_no_reference_duration(self) -> None:
        module = self.require_module("test_speed.benchmark.benchmark_speakers")

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            voice_path = root / "any.pth"
            output_dir = root / "embeddings"
            torch.save({"tensor": torch.ones(256)}, voice_path)
            model = RecordingVoiceModel()

            prepared = module.prepare_single_vector_voice(
                model,
                module.SingleVectorVoiceSource(voice_path),
                output_dir,
            )

            self.assertEqual(prepared.voice_ids, ["any"])
            self.assertEqual(prepared.reference_durations, [])
            self.assertEqual(prepared.source_names, ["any.pth"])
            self.assertEqual(set(model.loaded), {"any"})

    def test_raw_tensor_voice_needs_no_reference_duration(self) -> None:
        module = self.require_module("test_speed.benchmark.benchmark_speakers")

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            voice_path = root / "ljspeech.pth"
            output_dir = root / "embeddings"
            torch.save(torch.ones(1, 256), voice_path)
            model = RecordingVoiceModel()

            prepared = module.prepare_tensor_voice(
                model,
                module.TensorVoiceSource(voice_path),
                output_dir,
            )

            self.assertEqual(prepared.voice_ids, ["ljspeech"])
            self.assertEqual(prepared.reference_durations, [])
            self.assertEqual(model.loaded["ljspeech"].shape, (1, 256))

    def test_tensor_voice_directory_loads_all_selected_voices(self) -> None:
        module = self.require_module("test_speed.benchmark.benchmark_speakers")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            voices_dir = root / "voices"
            voices_dir.mkdir()
            for name in ("libri_f1", "libri_f2", "libri_m1"):
                torch.save(torch.ones(1, 256), voices_dir / f"{name}.pth")
            model = RecordingVoiceModel()
            prepared = module.prepare_tensor_voices(
                model, module.TensorVoiceDirectorySource(voices_dir), root / "out", 3, 1
            )
            self.assertEqual(set(prepared.voice_ids), set(model.loaded))
            self.assertEqual(set(prepared.voice_ids), {"libri_f1", "libri_f2", "libri_m1"})


class StyleNormTests(OptionalFeatureTest):
    def test_norms_cover_full_and_both_vector_halves(self) -> None:
        module = self.require_module("test_speed.benchmark.benchmark_style")
        measure = module.measure_style_norms
        vector = torch.zeros(256)
        vector[0:2] = torch.tensor([3.0, 4.0])
        vector[128:130] = torch.tensor([5.0, 12.0])

        norms = measure(RecordingVoiceModel({"voice": vector}), ["voice"])

        self.assertAlmostEqual(norms[0].full_norm, np.sqrt(194.0), places=6)
        self.assertAlmostEqual(norms[0].first_half_norm, 5.0)
        self.assertAlmostEqual(norms[0].second_half_norm, 13.0)

    def test_three_style_norm_plots_use_all_runs(self) -> None:
        module = self.require_module("test_speed.benchmark.benchmark_style")
        norm_type = module.StyleEmbeddingNorm
        write_plots = module.write_style_norm_plots
        metrics = [
            RequestMetric("a", "x", "x", 1, 7, 1, 0.05, 20.0, "a.wav"),
            RequestMetric("a", "y", "y", 1, 9, 1, 0.05, 20.0, "b.wav"),
            RequestMetric("b", "x", "x", 1, 7, 1, 0.05, 20.0, "c.wav"),
        ]
        norms = [
            norm_type("a", 1.0, 2.0, 3.0),
            norm_type("b", 4.0, 5.0, 6.0),
        ]

        with tempfile.TemporaryDirectory() as directory:
            paths = write_plots(metrics, norms, Path(directory))

            self.assertEqual(len(paths), 3)
            self.assertTrue(all(path.is_file() for path in paths))
            self.assertEqual(
                module.style_norm_coordinates(
                    metrics,
                    norms,
                    module.StyleNormPart.FULL,
                ),
                ([7, 9, 7], [1.0, 1.0, 4.0]),
            )

    def test_reports_include_all_three_style_norm_plots(self) -> None:
        module = self.require_module("test_speed.benchmark.benchmark_style")
        self.assertIn("style_norms", signature(write_reports).parameters)
        requests = [
            RequestMetric("v", "x", "x", 1, 7, 1, 0.05, 20.0, "a.wav")
        ]
        phonemes = [PhonemeMetric("a", 0.05, "v", "x")]
        references = [ReferenceDuration("v", 2.0)]
        norms = [module.StyleEmbeddingNorm("v", 1.0, 2.0, 3.0)]

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_reports(
                root,
                requests,
                phonemes,
                references,
                norms,
                ["v"],
                np.asarray([19.0, 20.0, 21.0]),
            )

            self.assertEqual(
                len(list((root / "summary").glob("style_embedding_*.png"))),
                3,
            )

    def test_reports_without_duration_omit_reference_plots(self) -> None:
        module = self.require_module("test_speed.benchmark.benchmark_style")
        requests = [
            RequestMetric("v", "x", "x", 1, 7, 1, 0.05, 20.0, "a.wav")
        ]
        phonemes = [PhonemeMetric("a", 0.05, "v", "x")]
        norms = [module.StyleEmbeddingNorm("v", 1.0, 2.0, 3.0)]

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            index = write_reports(
                root,
                requests,
                phonemes,
                [],
                norms,
                ["v"],
                np.asarray([19.0, 20.0, 21.0]),
            ).read_text()

            self.assertEqual(
                list((root / "summary").glob("reference_duration_*.png")),
                [],
            )
            self.assertNotIn("Reference length", index)


class RunnerTargetTests(unittest.TestCase):
    def test_all_selects_magda_and_agnieszka_outputs(self) -> None:
        runner = import_module("test_speed.benchmark.run_benchmark")
        select_targets = getattr(runner, "select_targets", None)
        self.assertIsNotNone(select_targets)
        assert select_targets is not None

        targets = select_targets("all")

        self.assertEqual(
            [target.name for target in targets],
            ["magda", "agnieszka", "olam", "vokan", "ljspeech", "libri", "styletts_finetune_epoch10"],
        )
        self.assertEqual(
            [target.results_dir.parent.name for target in targets],
            [
                "magda_original",
                "agnieszka",
                "olam",
                "vokan",
                "ljspeech",
                "libri",
                "styletts_finetune_epoch10",
            ],
        )

    def test_single_command_forwards_speaker_selection(self) -> None:
        script = Path("test_speed/run.sh").read_text()

        self.assertIn('python -m test_speed.benchmark.run_benchmark "$@"', script)
        self.assertIn("--with phonemizer", script)

    def test_olam_selects_only_olam_outputs(self) -> None:
        runner = import_module("test_speed.benchmark.run_benchmark")

        targets = runner.select_targets("olam")

        self.assertEqual([target.name for target in targets], ["olam"])
        self.assertEqual(targets[0].results_dir.parent.name, "olam")
        self.assertEqual(targets[0].voice_count, 1)

        finetune = runner.select_targets("styletts_finetune_epoch10")[0]
        self.assertEqual((finetune.voice_count, finetune.language), (10, "pl"))
        self.assertEqual((finetune.voice_source.archive_path.name, finetune.use_training_phonemes), ("backend_references.zip", True))

    def test_english_targets_use_english_phonemization(self) -> None:
        runner = import_module("test_speed.benchmark.run_benchmark")
        inference = import_module("test_speed.benchmark.benchmark_inference")

        targets = [
            runner.select_targets(name)[0]
            for name in ("vokan", "ljspeech", "libri")
        ]

        self.assertEqual([target.voice_count for target in targets], [1, 1, 3])
        self.assertTrue(all(target.passage.startswith("Early") for target in targets))
        self.assertEqual([target.runtime_engine for target in targets], ["torch"] * 3)
        self.assertEqual([target.language for target in targets], ["en-us"] * 3)
        self.assertIn("runtime_engine", signature(inference.load_model).parameters)


if __name__ == "__main__":
    unittest.main()
