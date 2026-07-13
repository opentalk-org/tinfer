from __future__ import annotations

from dataclasses import fields
from pathlib import Path
from typing import Any

import torch
from munch import Munch

from tinfer.core.request import AlignmentType
from tinfer.models.base.model import IntermediateRepresentation
from tinfer.models.chunked import ChunkedModel
from tinfer.models.impl.styletts2.alignment.alignment import StyleTTS2AlignmentParser
from tinfer.models.impl.styletts2.model.modules.config import ModelConfig
from tinfer.models.impl.styletts2.model.tensorrt_accelerator import StyleTTS2TensorRTAccelerator
from tinfer.models.impl.styletts2.voice.cache import VoiceCache
from tinfer.support.observability import get_logger
from .batch_audio import BatchAudioMixin
from .batch_style import BatchStyleMixin
from .batch_text import BatchTextMixin
from .inference_config import StyleTTS2Params
from .model_loading import ModelLoadingMixin
from .model_utils import timed_operation
from .phonemizer import StyleTTS2Phonemizer
from .text_config import TextConfig
from .voice_context import VoiceContextMixin

log = get_logger(__name__)


class StyleTTS2(
    ModelLoadingMixin,
    VoiceContextMixin,
    BatchTextMixin,
    BatchStyleMixin,
    BatchAudioMixin,
    ChunkedModel,
):
    def __init__(self, device: str = "cuda") -> None:
        self._loaded = False
        self._device = device
        self._sample_rate = 24000
        self._config: dict[str, Any] = {}
        self._model: Munch | None = None
        self._model_config: ModelConfig | None = None
        self._sampler = None
        self._phonemizer: StyleTTS2Phonemizer | None = None
        self._phonemizers: dict[str, StyleTTS2Phonemizer] = {}
        self._text_config: TextConfig | None = None
        self._default_language: str | None = None
        self._text_processing_pipeline = None
        self._voice_encoder: Any | None = None
        self._alignment_parser = StyleTTS2AlignmentParser()
        self._voice_cache = VoiceCache()
        self.max_batch_size = 10
        self._model_dir: Path | None = None

        self._compile_model_flag = False
        self._trt = StyleTTS2TensorRTAccelerator()
        self._max_styletts_tokens = 512
        self._end_trim_margin_ms = 500
    
    @torch.inference_mode()
    def generate_batch(
        self,
        texts: list[str],
        contexts: list[dict[str, Any] | None],
        params: list[dict[str, Any]],
        request_metadata: list[dict[str, Any]],
    ) -> list[IntermediateRepresentation]:
        with timed_operation("generate_batch"):
            self._validate_generation_ready()
            
            batch_size = len(texts)
            
            with timed_operation("prepare_voice_contexts"):
                self._prepare_voice_contexts(texts, contexts, batch_size)

            styletts2_params_list = []
            for i in range(batch_size):
                valid_params = {f.name: params[i][f.name] for f in fields(StyleTTS2Params) if f.name in params[i]}
                styletts2_params = StyleTTS2Params(**valid_params)
                styletts2_params_list.append(styletts2_params)
                metadata = request_metadata[i] if i < len(request_metadata) else {}
                log.debug(
                    "styletts2_inference_params",
                    request_id=metadata.get("request_id"),
                    chunk_index=metadata.get("chunk_index"),
                    alpha=styletts2_params.alpha,
                    beta=styletts2_params.beta,
                )

            if self._phonemizer is None:
                raise RuntimeError("Phonemizer not initialized")

            windowed_results = self._generate_token_windows(
                texts,
                contexts,
                params,
                request_metadata,
                styletts2_params_list,
            )
            if windowed_results is not None:
                return windowed_results
            
            alignment_type = request_metadata[0]["alignment_type"]
            if isinstance(alignment_type, str):
                alignment_type = AlignmentType(alignment_type)
            
            use_diffusion_any = any(p.use_diffusion for p in styletts2_params_list)
            
            if use_diffusion_any and self._sampler is None and not self._trt.diffusion_enabled:
                self._build_sampler()
            
            with timed_operation("prepare_voice_tensors"):
                ref_s_batch = self._prepare_voice_tensors(contexts, batch_size)
            
            with timed_operation("prepare_previous_style_vectors"):
                prev_s_list = self._prepare_previous_style_vectors(contexts, batch_size)
            
            with timed_operation("process_texts"):
                all_tokens, all_tokens_for_alignment, all_phonemized_texts_for_alignment, original_texts = self._process_texts(texts, alignment_type, styletts2_params_list)
            
            with timed_operation("prepare_token_tensors"):
                tokens_tensor, input_lengths_tensor, text_mask = self._prepare_token_tensors(all_tokens)
            
            input_lengths = [len(tokens) for tokens in all_tokens]
            
            out, all_pred_aln_trg, all_actual_lengths, s_pred = self._run_model_forward(
                tokens_tensor,
                input_lengths_tensor,
                text_mask,
                ref_s_batch,
                prev_s_list,
                styletts2_params_list,
                batch_size,
                input_lengths,
            )
            
            with timed_operation("post_process_results"):
                results = self._post_process_results(
                    out,
                    all_pred_aln_trg,
                    all_tokens_for_alignment,
                    all_phonemized_texts_for_alignment,
                    original_texts,
                    all_actual_lengths,
                    contexts,
                    params,
                    alignment_type,
                    styletts2_params_list,
                    ref_s_batch,
                    s_pred,
                    batch_size,
                )
            
            return results
    
    
