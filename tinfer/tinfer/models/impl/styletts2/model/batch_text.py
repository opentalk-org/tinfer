from __future__ import annotations

import re
from typing import Any

import torch

from tinfer.core.request import AlignmentType
from tinfer.models.base.model import IntermediateRepresentation
from .inference_config import StyleTTS2Params
from .phonemizer import StyleTTS2Phonemizer
from .speed_correction import baseline_speed_corrected_for_request
from .model_utils import length_to_mask

class BatchTextMixin:
    def _get_phonemizer(self, language: str | None) -> StyleTTS2Phonemizer:
        lang = language or self._default_language
        assert self._text_config is not None, "loaded model requires text config"
        if lang not in self._text_config.supported_languages:
            raise ValueError(f"language '{lang}' is not supported by this model")
        if lang not in self._phonemizers:
            self._phonemizers[lang] = StyleTTS2Phonemizer(
                language=lang,
                symbols=self._text_config.symbols,
            )
        return self._phonemizers[lang]

    def _process_texts(self, texts: list[str], alignment_type: AlignmentType, styletts2_params_list: list[StyleTTS2Params]) -> tuple[list[list[int]], list[list[int]], list, list[str]]:
        all_tokens = []
        all_tokens_for_alignment = []
        all_phonemized_texts_for_alignment = []
        original_texts = texts.copy()
        
        for i, text in enumerate(texts):
            styletts2_params = styletts2_params_list[i]
            phonemizer = self._get_phonemizer(styletts2_params.language)

            if (
                self._text_processing_pipeline is not None
                and styletts2_params.apply_text_normalization != "off"
            ):
                text = self._text_processing_pipeline.process(text)

            if not styletts2_params.phonemized:
                processed_text, word_phoneme_data = phonemizer.process_text_with_original_spans(text)
                tokens_without_bos = phonemizer.tokenize(processed_text)
            else:
                tokens_without_bos = phonemizer.tokenize(text)
                word_phoneme_data = None

            styletts2_params.speed = baseline_speed_corrected_for_request(
                styletts2_params.speed,
                len(tokens_without_bos),
            )
            tokens_for_alignment = tokens_without_bos.copy()
            tokens = tokens_without_bos.copy()
            tokens.insert(0, 0)
            all_tokens.append(tokens)
            all_tokens_for_alignment.append(tokens_for_alignment.copy())
            all_phonemized_texts_for_alignment.append(word_phoneme_data)
        
        return all_tokens, all_tokens_for_alignment, all_phonemized_texts_for_alignment, original_texts
    
    def _prepare_token_tensors(self, all_tokens: list[list[int]]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        max_token_len = max(len(tokens) for tokens in all_tokens)
        
        padded_tokens = []
        input_lengths = []
        for tokens in all_tokens:
            padded = tokens + [0] * (max_token_len - len(tokens))
            padded_tokens.append(padded)
            input_lengths.append(len(tokens))
        
        tokens_tensor = torch.LongTensor(padded_tokens).to(self._device)
        input_lengths_tensor = torch.LongTensor(input_lengths).to(self._device)
        text_mask = length_to_mask(input_lengths_tensor).to(self._device)
        
        return tokens_tensor, input_lengths_tensor, text_mask
    
    def _pad_to_batch_size(self, tensor: torch.Tensor, batch_size: int) -> tuple[torch.Tensor, bool]:
        if batch_size >= self.max_batch_size:
            return tensor, False
        padding_size = self.max_batch_size - batch_size
        last_element = tensor[-1:].expand(padding_size, *tensor.shape[1:])
        padded = torch.cat([tensor, last_element], dim=0)
        return padded, True
    
    def _pad_to_multiple(self, tensor: torch.Tensor, multiple: int, dim: int = -1) -> torch.Tensor:
        if dim < 0:
            dim = len(tensor.shape) + dim
        seq_len = tensor.shape[dim]
        remainder = seq_len % multiple
        if remainder == 0:
            return tensor
        padding_size = multiple - remainder
        pad_shape = list(tensor.shape)
        pad_shape[dim] = padding_size
        last_slice = tensor.select(dim, seq_len - 1).unsqueeze(dim)
        padding = last_slice.expand(*pad_shape)
        return torch.cat([tensor, padding], dim=dim)

    def _text_token_count(self, text: str, styletts2_params: StyleTTS2Params) -> int:
        phonemizer = self._get_phonemizer(styletts2_params.language)
        if styletts2_params.phonemized:
            tokens = phonemizer.tokenize(text)
        else:
            processed_text, _ = phonemizer.process_text_with_original_spans(text)
            tokens = phonemizer.tokenize(processed_text)
        return len(tokens) + 1

    def _split_text_to_token_windows(self, text: str, styletts2_params: StyleTTS2Params) -> list[str]:
        return self._split_text_to_token_windows_recursive(
            text,
            styletts2_params,
            [r"(?<=[.!?]) +", r"(?<=[,;:]) +", " "],
        )

    def _split_text_to_token_windows_recursive(
        self,
        text: str,
        styletts2_params: StyleTTS2Params,
        separators: list[str],
    ) -> list[str]:
        if self._text_token_count(text, styletts2_params) <= self._max_styletts_tokens:
            return [text]
        if not separators:
            return self._split_text_to_token_windows_by_char(text, styletts2_params)

        parts = self._split_keep_separator(text, separators[0])
        if len(parts) <= 1:
            return self._split_text_to_token_windows_recursive(text, styletts2_params, separators[1:])

        chunks = []
        current = ""
        for part in parts:
            candidate = current + part if current else part
            if current and self._text_token_count(candidate, styletts2_params) > self._max_styletts_tokens:
                chunks.extend(
                    self._split_text_to_token_windows_recursive(
                        current.rstrip(),
                        styletts2_params,
                        separators[1:],
                    )
                )
                current = part
            else:
                current = candidate
        if current.strip():
            chunks.extend(
                self._split_text_to_token_windows_recursive(
                    current.rstrip(),
                    styletts2_params,
                    separators[1:],
                )
            )
        return chunks

    def _split_keep_separator(self, text: str, separator: str) -> list[str]:
        if separator == " ":
            return [part for part in re.findall(r"\S+\s*", text) if part]

        parts = re.split(f"({separator})", text)
        merged = []
        i = 0
        while i < len(parts):
            part = parts[i]
            if not part:
                i += 1
                continue
            if i + 1 < len(parts):
                part += parts[i + 1]
                i += 2
            else:
                i += 1
            merged.append(part)
        return merged

    def _split_text_to_token_windows_by_char(
        self,
        text: str,
        styletts2_params: StyleTTS2Params,
    ) -> list[str]:
        chunks = []
        current = ""
        for char in text:
            candidate = current + char
            if current and self._text_token_count(candidate, styletts2_params) > self._max_styletts_tokens:
                chunks.append(current)
                current = char
            else:
                current = candidate
        if current.strip():
            chunks.append(current)
        return chunks

    def _generate_token_windows(
        self,
        texts: list[str],
        contexts: list[dict[str, Any] | None],
        params: list[dict[str, Any]],
        request_metadata: list[dict[str, Any]],
        styletts2_params_list: list[StyleTTS2Params],
    ) -> list[IntermediateRepresentation] | None:
        windows_by_text = [
            self._split_text_to_token_windows(text, styletts2_params)
            for text, styletts2_params in zip(texts, styletts2_params_list)
        ]
        if all(len(windows) == 1 for windows in windows_by_text):
            return None

        merged_results = []
        for text, windows, context, param, metadata in zip(texts, windows_by_text, contexts, params, request_metadata):
            window_context = context if context is not None else {}
            window_results = [
                self.generate_batch([window], [window_context], [param], [metadata])[0]
                for window in windows
            ]
            merged_results.append(self._merge_window_results(window_results, text))
        return merged_results
