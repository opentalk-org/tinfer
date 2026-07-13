from types import SimpleNamespace
import unittest

import torch

from tinfer.core.request import AlignmentType
from tinfer.models.impl.styletts2.model.inference_config import StyleTTS2Params
from tinfer.models.impl.styletts2.model.model import StyleTTS2


class DecoderOutputLengthTests(unittest.TestCase):
    def test_hifigan_post_processing_keeps_backbone_upsampling(self):
        model = StyleTTS2(device="cpu")
        model._model_config = SimpleNamespace(
            preprocess=SimpleNamespace(hop_length=300),
            decoder=SimpleNamespace(type="hifigan", upsample_rates=[10, 5, 3, 2]),
        )
        model._sample_rate = 24_000
        output = torch.zeros((1, 1, 1_200))

        results = model._post_process_results(
            out=output,
            all_pred_aln_trg=[None],
            all_tokens_for_alignment=[[]],
            all_phonemized_texts_for_alignment=[None],
            original_texts=["test"],
            all_actual_lengths=[2],
            contexts=[None],
            params=[{}],
            alignment_type=AlignmentType.NONE,
            styletts2_params_list=[StyleTTS2Params(use_diffusion=False)],
            ref_s_batch=torch.zeros((1, 256)),
            s_pred=None,
            batch_size=1,
        )

        self.assertEqual(len(results[0].data), 1_100)


if __name__ == "__main__":
    unittest.main()
