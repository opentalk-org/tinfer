from __future__ import annotations

from typing import Any

import numpy as np
import torch

from tinfer.core.request import AlignmentItem, AlignmentType
from tinfer.models.base.model import IntermediateRepresentation
from tinfer.support.observability import get_logger
from .inference_config import StyleTTS2Params
from .model_utils import _alignment_items_for_debug_log

log = get_logger(__name__)

class BatchAudioMixin:
    def _post_process_results(
        self,
        out: torch.Tensor,
        all_pred_aln_trg: list,
        all_tokens_for_alignment: list[list[int]],
        all_phonemized_texts_for_alignment: list,
        original_texts: list[str],
        all_actual_lengths: list[int],
        contexts: list[dict[str, Any] | None],
        params: list[dict[str, Any]],
        alignment_type: AlignmentType,
        styletts2_params_list: list[StyleTTS2Params],
        ref_s_batch: torch.Tensor,
        s_pred: torch.Tensor | None,
        batch_size: int,
    ) -> list[IntermediateRepresentation]:
        results = []
        hop_length = self._model_config.preprocess.hop_length
        decoder_hop_length = hop_length
        decoder_type = getattr(self._model_config.decoder, "type", None)
        if decoder_type == "istftnet":
            decoder_hop_length = int(np.prod(self._model_config.decoder.upsample_rates) * self._model_config.decoder.gen_istft_hop_size * 2)
        elif decoder_type == "hifigan":
            decoder_hop_length = int(np.prod(self._model_config.decoder.upsample_rates) * 2)
        out_cpu = out[:batch_size].detach().cpu().numpy()
        ref_s_batch_cpu = ref_s_batch[:batch_size].detach().cpu().numpy()
        s_pred_cpu = s_pred[:batch_size].detach().cpu().numpy() if s_pred is not None else None
        
        for b in range(batch_size):
            use_diffusion = styletts2_params_list[b].use_diffusion
            actual_mel_frames = all_actual_lengths[b]
            audio = out_cpu[b]
            
            if audio.ndim > 1:
                audio = audio.squeeze()

            expected_audio_length = int(actual_mel_frames * decoder_hop_length)
            if 0 < expected_audio_length < len(audio):
                audio = audio[:expected_audio_length]
            if len(audio) > 100:
                audio = audio[:-100]

            sample_rate = self._sample_rate
            target_alignment_type = alignment_type

            if target_alignment_type == AlignmentType.NONE:
                final_alignments = []
            else:
                phoneme_alignments = self._alignment_parser.parse_from_pred_aln_trg(
                    all_pred_aln_trg[b],
                    all_tokens_for_alignment[b],
                    original_texts[b],
                    self._phonemizer.detokenize(all_tokens_for_alignment[b]),
                    sample_rate=self._sample_rate,
                    hop_length=hop_length,
                    actual_audio_length=len(audio),
                )

                if target_alignment_type == AlignmentType.WORD:
                    word_phoneme_data = all_phonemized_texts_for_alignment[b]
                    word_alignments = self._alignment_parser.convert_to_word(
                        phoneme_alignments,
                        original_texts[b],
                        word_phoneme_data,
                    )
                    final_alignments = word_alignments
                elif target_alignment_type == AlignmentType.CHAR:
                    word_phoneme_data = all_phonemized_texts_for_alignment[b]
                    char_alignments = self._alignment_parser.convert_to_char(
                        phoneme_alignments,
                        original_texts[b],
                        word_phoneme_data,
                    )
                    final_alignments = char_alignments
                else:
                    final_alignments = phoneme_alignments

            if target_alignment_type in (AlignmentType.CHAR, AlignmentType.WORD):
                assert "".join(item.item for item in final_alignments) == original_texts[b]
                log.debug(
                    "alignment_processed",
                    text=original_texts[b],
                    alignment_type=target_alignment_type.value,
                    alignments=_alignment_items_for_debug_log(final_alignments),
                )
            
            current_style_vector = s_pred_cpu[b] if use_diffusion and s_pred_cpu is not None else ref_s_batch_cpu[b]
            
            if b < len(contexts) and contexts[b] is not None:
                contexts[b]["previous_style_vector"] = current_style_vector.copy()
            
            metadata = {
                "text": original_texts[b],
                "sample_rate": sample_rate,
                "style_vector": current_style_vector,
                "alignment_type": alignment_type.value,
                "word_alignments": final_alignments,
            }
            
            results.append(IntermediateRepresentation(
                data=audio,
                sample_rate=sample_rate,
                metadata=metadata,
            ))
        
        return results

    def _merge_window_results(self, results: list[IntermediateRepresentation], original_text: str) -> IntermediateRepresentation:
        if len(results) == 1:
            return results[0]
        sample_rate = results[0].sample_rate
        audio_parts = [np.asarray(result.data) for result in results]
        audio = np.concatenate(audio_parts, axis=-1)
        metadata = dict(results[-1].metadata)
        merged_alignments: list[AlignmentItem] = []
        time_offset_ms = 0
        search_start = 0

        for result, audio_part in zip(results, audio_parts):
            window_text = result.metadata.get("text", "")
            if isinstance(window_text, str) and window_text:
                char_offset = original_text.find(window_text, search_start)
                if char_offset < 0:
                    char_offset = search_start
                search_start = char_offset + len(window_text)
            else:
                char_offset = search_start

            for item in result.metadata.get("word_alignments", []) or []:
                merged_alignments.append(
                    AlignmentItem(
                        item=item.item,
                        char_start=item.char_start + char_offset,
                        char_end=item.char_end + char_offset,
                        start_ms=item.start_ms + time_offset_ms,
                        end_ms=item.end_ms + time_offset_ms,
                    )
                )

            time_offset_ms += int(round((audio_part.shape[-1] / sample_rate) * 1000.0)) if sample_rate else 0

        metadata["text"] = original_text
        metadata["window_count"] = len(results)
        metadata["word_alignments"] = merged_alignments
        return IntermediateRepresentation(data=audio, sample_rate=sample_rate, metadata=metadata)
