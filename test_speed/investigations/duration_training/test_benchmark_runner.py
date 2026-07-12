import unittest

from test_speed.investigations.duration_training.benchmark_runner import FINETUNED_TARGET


class FineTunedBenchmarkTests(unittest.TestCase):
    def test_uses_finetuned_model_and_separate_profile_folders(self) -> None:
        self.assertEqual(FINETUNED_TARGET.name, "magda_duration_finetune")
        self.assertEqual(
            FINETUNED_TARGET.model_path.name,
            "model.pth",
        )
        self.assertEqual(
            FINETUNED_TARGET.model_path.parent.name,
            "magda_duration_consistency",
        )
        self.assertEqual(FINETUNED_TARGET.results_dir.name, "diffusion")
        self.assertEqual(
            FINETUNED_TARGET.no_diffusion_results_dir.name,
            "no_diffusion",
        )


if __name__ == "__main__":
    unittest.main()
