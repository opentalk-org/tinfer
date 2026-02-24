from __future__ import annotations
from typing import Any
import numpy as np

from tinfer.core.request import AlignmentItem, AlignmentType
from tinfer.models.base.alignment import AlignmentParser
from tinfer.models.impl.styletts2.alignment.converter import AlignmentConverter


class StyleTTS2AlignmentParser(AlignmentParser):
    def __init__(self):
        self._converter = AlignmentConverter()
    
    def get_native_alignment_type(self) -> AlignmentType:
        """StyleTTS2 natively provides phoneme-level alignments."""
        return AlignmentType.PHONEME
    
    def parse_from_pred_aln_trg(
        self,
        pred_aln_trg: np.ndarray,
        tokens: list[int],
        original_text: str,
        phonemized_text: str,
        sample_rate: int = 24000,
        hop_length: int = 300,
        actual_audio_length: int | None = None,
    ) -> list[AlignmentItem]:
        if pred_aln_trg.shape[0] == 0 or pred_aln_trg.shape[1] == 0:
            return []
        
        num_tokens = pred_aln_trg.shape[0]
        num_frames = pred_aln_trg.shape[1]

        ms_per_mel_frame = (hop_length / sample_rate) * 1000
        ms_per_mel_frame = 1000 / 40
        
        token_durations_frames = pred_aln_trg.sum(axis=1)
        token_start_frames = np.array([
            np.argmax(pred_aln_trg[i, :] > 0) - 1 if np.any(pred_aln_trg[i, :] > 0) else 0
            for i in range(num_tokens)
        ])
        
        phonemized_chars = list(phonemized_text)
        
        alignments = []
        for token_idx in range(num_tokens):
            if token_idx >= len(phonemized_chars):
                continue
            
            start_frame = token_start_frames[token_idx]
            duration_frames = token_durations_frames[token_idx]
            
            start_ms = int(start_frame * ms_per_mel_frame)
            duration_ms = int(duration_frames * ms_per_mel_frame)
            end_ms = start_ms + duration_ms
            
            char = phonemized_chars[token_idx] if token_idx < len(phonemized_chars) else ""
            
            alignments.append(AlignmentItem(
                item=char,
                start_ms=start_ms,
                end_ms=end_ms,
                char_start=token_idx,
                char_end=token_idx + 1,
            ))
        
        return alignments

    
    def convert_to_word(
        self,
        phoneme_alignments: list[AlignmentItem],
        original_text: str,
        phonemized_text_with_sep: str | None = None,
    ) -> list[AlignmentItem]:
        return self._converter.phoneme_to_word(
            phoneme_alignments,
            original_text,
            phonemized_text=phonemized_text_with_sep,
        )
    
    def convert_to_char(
        self,
        phoneme_alignments: list[AlignmentItem],
        original_text: str,
        phonemized_text: str | None = None,
    ) -> list[AlignmentItem]:

        return self._converter.phoneme_to_char(
            phoneme_alignments,
            original_text,
            phonemized_text=phonemized_text,
        )


