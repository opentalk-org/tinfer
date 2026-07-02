from __future__ import annotations
import torch
import numpy as np
from pathlib import Path
from typing import Any
from dataclasses import asdict, fields
from munch import Munch
import importlib
import warnings
import librosa
import time
import os
import json
import re
from contextlib import contextmanager

from tinfer.models.chunked import ChunkedModel
from tinfer.models.base.model import IntermediateRepresentation
from tinfer.core.request import AlignmentItem, AlignmentType
from tinfer.models.impl.styletts2.alignment.alignment import StyleTTS2AlignmentParser
from tinfer.models.impl.styletts2.model.modules.diffusion.sampler import DiffusionSampler, ADPM2Sampler, KarrasSchedule
from tinfer.models.impl.styletts2.model.modules.load_utils import load_model, load_model_from_state, get_model_state_dict
from tinfer.models.impl.styletts2.model.modules.config import ModelConfig
from tinfer.models.impl.styletts2.model.tensorrt_accelerator import (
    StyleTTS2TensorRTAccelerator,
)
from .phonemizer import StyleTTS2Phonemizer
from .inference_config import StyleTTS2Params
from tinfer.models.impl.styletts2.voice.cache import VoiceCache


def length_to_mask(lengths):
    mask = torch.arange(lengths.max()).unsqueeze(0).expand(lengths.shape[0], -1).type_as(lengths)
    mask = torch.gt(mask + 1, lengths.unsqueeze(1))
    return mask


def _find_leading_audio_samples(audio: np.ndarray, sample_rate: int) -> int:
    if audio.size == 0 or sample_rate <= 0:
        return 0

    frame_size = max(1, int(sample_rate * 0.02))
    starts = np.arange(0, audio.size, frame_size)
    if starts.size == 0:
        return 0

    audio64 = audio.astype(np.float64, copy=False)
    squared = audio64 * audio64
    frame_energy = np.add.reduceat(squared, starts)
    frame_lengths = np.minimum(frame_size, audio.size - starts)
    rms = np.sqrt(frame_energy / frame_lengths)
    threshold = max(0.01, float(np.percentile(rms, 80)) * 0.1)
    above = np.where(rms > threshold)[0]
    if len(above) == 0:
        return 0

    first_frame = int(above[0])
    trim_samples = first_frame * frame_size
    return trim_samples if trim_samples >= int(sample_rate * 0.05) else 0


def _trim_leading_silence_and_shift_alignments(
    audio: np.ndarray,
    sample_rate: int,
    alignments: list[AlignmentItem],
) -> tuple[np.ndarray, list[AlignmentItem]]:
    trim_samples = _find_leading_audio_samples(audio, sample_rate)
    if trim_samples <= 0:
        return audio, alignments

    trim_ms = int(round(trim_samples / sample_rate * 1000.0))
    shifted = [
        AlignmentItem(
            item=item.item,
            char_start=item.char_start,
            char_end=item.char_end,
            start_ms=max(0, item.start_ms - trim_ms),
            end_ms=max(0, item.end_ms - trim_ms),
        )
        for item in alignments
    ]
    return audio[trim_samples:], shifted


@contextmanager
def timed_operation(name: str):
    profile_enabled = bool(os.getenv("TINFER_PROFILE"))
    if profile_enabled and torch.cuda.is_available():
        torch.cuda.synchronize()
    start_time = time.perf_counter()
    try:
        yield
    finally:
        if profile_enabled:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            print(
                "TINFER_PROFILE "
                + json.dumps(
                    {
                        "scope": "model_stage",
                        "stage": name,
                        "elapsed_ms": elapsed_ms,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

class StyleTTS2(ChunkedModel):
    def __init__(self, device: str = "cuda") -> None:
        self._loaded = False
        self._device = device
        self._sample_rate = 24000
        self._config: dict[str, Any] = {}
        self._model: Munch | None = None
        self._model_config: ModelConfig | None = None
        self._sampler = None
        self._phonemizer: StyleTTS2Phonemizer | None = None
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
    
    def _initialize_from_config(self, config: dict[str, Any], device: str) -> None:
        self._config = config
        self._sample_rate = config.get("sample_rate", 24000)
        self.max_batch_size = config.get("max_batch_size", 10)
        
        load_style_encoder = config.get("load_style_encoder", True)
        language = config.get("language", "pl")
        
        self._model = Munch({key: self._model[key].to(self._device) for key in self._model})
        _ = [self._model[key].eval() for key in self._model]
        
        self._phonemizer = StyleTTS2Phonemizer(language=language)
        self._text_processing_pipeline = config.get("text_processing_pipeline")
        
        if load_style_encoder:
            voice_encoder_module = importlib.import_module("tinfer.models.impl.styletts2.voice.encoder")
            self._voice_encoder = voice_encoder_module.StyleTTS2VoiceEncoder(
                model=self._model,
                device="cpu",
                sample_rate=self._sample_rate
            )

        self.to(device)
        self._trt = StyleTTS2TensorRTAccelerator.from_runtime_config(
            config,
            self._model_dir or Path("."),
            self._model,
            self._model_config,
        )
        
        self._loaded = True

    def _compile_model(self) -> None:
        # pass
        # pass
        # self._model['decoder'].generator.compile_resblocks()
        print("Compiling decoder...")
        self._model['decoder'].generator._forward_compiled = torch.compile(
            self._model['decoder'].generator._forward_compiled,
            mode='reduce-overhead',
            dynamic=True
        )
        print("Decoder compiled")

        #self._model['decoder'].generator = torch.compile(self._model['decoder'].generator, mode='reduce-overhead')

    def to(self, device: str) -> None:
        self._device = device
        self._model = Munch({key: self._model[key].to(device) for key in self._model})
        _ = [self._model[key].eval() for key in self._model]
        if self._voice_encoder is not None:
            self._voice_encoder.to(device)

    def load(
        self,
        path: str,
        voices_folder: str | None = None,
        device="cuda",
        compile_model: bool = False,
        load_style_encoder: bool | None = None,
        runtime_engine: str | None = None,
    ) -> None:
        model_path = Path(path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")
        self._model_dir = model_path.parent
        
        try:
            model_saved = torch.load(str(model_path), map_location='cpu', weights_only=True)
        except Exception as e:
            raise ValueError(f"Error loading model from {path}: {e}, please check if the model isn't corrupted.")
        
        config = dict(model_saved.get('runtime_config', {}))
        if runtime_engine is not None:
            config["engine"] = runtime_engine
        if load_style_encoder is None:
            load_style_encoder = config.get("load_style_encoder", True)
        
        self._model, self._model_config = load_model(str(model_path), load_style_encoder)
        self._initialize_from_config(config, device)
        
        if voices_folder is not None:
            self.load_voices_from_folder(voices_folder)

        if compile_model:
            self._compile_model_flag = True
            self._compile_model()

    def get_state(self) -> dict[str, Any]:
        if not self._loaded:
            raise RuntimeError("Model not loaded. Call load() first.")
        if self._model is None or self._model_config is None:
            raise RuntimeError("Model not initialized")
        
        net_state_dicts = {}
        for key in self._model:
            state_dict = self._model[key].state_dict()
            cpu_state_dict = {k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in state_dict.items()}
            net_state_dicts[key] = cpu_state_dict
        
        config_dict = asdict(self._model_config) if hasattr(self._model_config, '__dataclass_fields__') else self._model_config
        
        state_dict = {
            'config': config_dict,
            'net': net_state_dicts,
        }
        if self._config:
            state_dict['runtime_config'] = self._config
        
        return state_dict

    def load_from_state(self, state: dict[str, Any], device: str = "cuda") -> None:
        config = state.get('runtime_config', {})
        load_style_encoder = config.get("load_style_encoder", True)
        
        self._model, self._model_config = load_model_from_state(state, load_style_encoder)
        self._initialize_from_config(config, device)
        self._compile_model()

    def load_voice_from_file(self, voice_id: str, file_path: str) -> None:
        if not self._loaded:
            raise RuntimeError("Model not loaded. Call load() first.")
        
        voice_path = Path(file_path)
        if not voice_path.exists():
            raise FileNotFoundError(f"Voice file not found: {file_path}")
        
        voice_data = torch.load(str(voice_path), map_location='cpu', weights_only=True)
        
        voice_vector = voice_data['voice_vector']
        
        if not isinstance(voice_vector, torch.Tensor):
            raise ValueError(f"Extracted voice data from {voice_path} is not a torch.Tensor, got {type(voice_vector)}")
        
        voice_vector = voice_vector.to(self._device)
        self._voice_cache.put(voice_id, voice_vector)

    def load_voice_from_audio(self, voice_id: str, audio_path: str, sample_rate: int | None = None) -> None:
        if not self._loaded:
            raise RuntimeError("Model not loaded. Call load() first.")
        
        if self._voice_encoder is None:
            raise RuntimeError("Voice encoder not available. Model must be loaded with load_style_encoder=True")
        
        audio_path_obj = Path(audio_path)
        if not audio_path_obj.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        if sample_rate is None:
            sample_rate = self._sample_rate
        
        waveform, sr = librosa.load(str(audio_path), sr=sample_rate)
        voice_vector = self._voice_encoder.compute_style_from_waveform(waveform, sr)
        
        self._voice_cache.put(voice_id, voice_vector)

    def load_voice_from_vector(self, voice_id: str, voice_vector: torch.Tensor) -> None:
        if not self._loaded:
            raise RuntimeError("Model not loaded. Call load() first.")
        
        if not isinstance(voice_vector, torch.Tensor):
            raise ValueError(f"voice_vector must be a torch.Tensor, got {type(voice_vector)}")
        
        voice_vector = voice_vector.to(self._device)
        self._voice_cache.put(voice_id, voice_vector)

    def load_voices_from_folder(self, voices_folder: str) -> None:
        if not self._loaded:
            raise RuntimeError("Model not loaded. Call load() first.")
        
        voices_path = Path(voices_folder)
        if not voices_path.exists():
            raise ValueError(f"Voices folder does not exist: {voices_folder}")
        
        if not voices_path.is_dir():
            raise ValueError(f"Voices path is not a directory: {voices_folder}")
        
        self._voice_cache.load(str(voices_path))
        
        if self._voice_encoder is not None:
            audio_extensions = {'.wav', '.mp3', '.flac', '.ogg', '.m4a'}
            
            for audio_path in voices_path.iterdir():
                if audio_path.suffix.lower() in audio_extensions:
                    voice_id = audio_path.stem
                    if voice_id not in self._voice_cache._cache:
                        try:
                            waveform, sr = librosa.load(str(audio_path), sr=self._sample_rate)
                            voice_vector = self._voice_encoder.compute_style_from_waveform(waveform, sr)
                            self._voice_cache.put(voice_id, voice_vector)
                        except Exception as e:
                            warnings.warn(f"Failed to load voice from audio file {audio_path}: {e}")

    def get_voice(self, voice_id: str) -> torch.Tensor:
        return self._voice_cache.get(voice_id)

    def has_voice(self, voice_id: str) -> bool:
        return voice_id in self._voice_cache._cache

    def list_voices(self) -> list[str]:
        return list(self._voice_cache._cache.keys())

    def clear_voices(self) -> None:
        self._voice_cache.clear()

    @property
    def device(self) -> str:
        return self._device
    
    @property
    def supports_streaming(self) -> bool:
        return False

    @property
    def supports_parallel_chunks(self) -> bool:
        return False

    @property
    def supports_parallel_sequential_chunks(self) -> bool:
        return False
    
    def supports_word_alignment(self) -> bool:
        return True

    def _build_sampler(self, sampler_config: dict[str, Any] | None = None):
        if self._sampler is not None:
            return
        
        if sampler_config is None:
            sampler_config = {
                'sampler': ADPM2Sampler(),
                'sigma_schedule': KarrasSchedule(sigma_min=0.0001, sigma_max=3.0, rho=9.0),
                'clamp': False
            }
        
        if self._model is None:
            raise ValueError("Model must be loaded before building sampler")
        
        self._sampler = DiffusionSampler(
            self._model.diffusion.diffusion,
            sampler=sampler_config.get('sampler', ADPM2Sampler()),
            sigma_schedule=sampler_config.get('sigma_schedule', 
                KarrasSchedule(sigma_min=0.0001, sigma_max=3.0, rho=9.0)),
            clamp=sampler_config.get('clamp', False)
        )
        if self._compile_model_flag:
            self._sampler.sampler.forward = torch.compile(
                self._sampler.sampler.forward,
                mode='reduce-overhead',
            )
    
    def _validate_generation_ready(self) -> None:
        if not self._loaded:
            raise RuntimeError("Model not loaded. Call load() first.")
        
        if self._model is None:
            raise RuntimeError("Model not initialized")
    
    def _prepare_voice_contexts(self, contexts: list[dict[str, Any] | None], batch_size: int) -> None:
        for i in range(batch_size):
            context = contexts[i] if i < len(contexts) and contexts[i] is not None else {}
            
            if context.get("reset_voice", False):
                if "base_voice" in context:
                    context["updated_voice"] = context["base_voice"].copy() if isinstance(context["base_voice"], np.ndarray) else context["base_voice"].clone()
                context["reset_voice"] = False
            
            voice_id = context.get("voice_id")
            if voice_id is not None and "base_voice" not in context:
                if not self.has_voice(voice_id):
                    raise ValueError(f"Voice ID '{voice_id}' not found in cache. Available voices: {self.list_voices()}")
                
                base_voice_tensor = self.get_voice(voice_id)
                base_voice_np = base_voice_tensor.cpu().numpy() if isinstance(base_voice_tensor, torch.Tensor) else base_voice_tensor
                context["base_voice"] = base_voice_np.copy() if isinstance(base_voice_np, np.ndarray) else base_voice_np
                context["updated_voice"] = base_voice_np.copy() if isinstance(base_voice_np, np.ndarray) else base_voice_np
    
    def _prepare_voice_tensors(self, contexts: list[dict[str, Any] | None], batch_size: int) -> torch.Tensor:
        ref_s_list = []
        for i in range(batch_size):
            context = contexts[i] if i < len(contexts) and contexts[i] is not None else {}
            
            if "updated_voice" in context:
                ref_s = context["updated_voice"]
            elif "base_voice" in context:
                ref_s = context["base_voice"]
            elif "style_vector" in context:
                ref_s = context["style_vector"]
            else:
                raise ValueError(f"Voice conditioning required for StyleTTS2 (request {i}). Provide voice_id, updated_voice, base_voice, or style_vector in context.")
            
            ref_s = torch.from_numpy(ref_s).to(self._device)
            if ref_s.dim() == 1:
                ref_s = ref_s.unsqueeze(0)
            ref_s_list.append(ref_s)
        
        ref_s_batch = torch.cat(ref_s_list, dim=0)
        return ref_s_batch
    
    def _prepare_previous_style_vectors(self, contexts: list[dict[str, Any] | None], batch_size: int) -> list:
        prev_s_list = []
        for i in range(batch_size):
            if "style_vector" in contexts[i]:
                prev_s = contexts[i]["style_vector"]
                prev_s = torch.from_numpy(prev_s).to(self._device)
                if prev_s.dim() == 1:
                    prev_s = prev_s.unsqueeze(0)
                prev_s_list.append(prev_s)
            else:
                prev_s_list.append(None)
        
        return prev_s_list
    
    def _process_texts(self, texts: list[str], alignment_type: AlignmentType, styletts2_params_list: list[StyleTTS2Params]) -> tuple[list[list[int]], list[list[int]], list, list[str]]:
        all_tokens = []
        all_tokens_for_alignment = []
        all_phonemized_texts_for_alignment = []
        original_texts = texts.copy()
        
        for i, text in enumerate(texts):
            styletts2_params = styletts2_params_list[i]
            
            if self._text_processing_pipeline is not None:
                text = self._text_processing_pipeline.process(text)

            if not styletts2_params.phonemized:
                processed_text, word_phoneme_data = self._phonemizer.process_text_with_original_spans(text)
                tokens_without_bos = self._phonemizer.tokenize(processed_text)
            else:
                tokens_without_bos = self._phonemizer.tokenize(text)
                word_phoneme_data = None

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

    def _run_model_forward(
        self,
        tokens_tensor: torch.Tensor,
        input_lengths_tensor: torch.Tensor,
        text_mask: torch.Tensor,
        ref_s_batch: torch.Tensor,
        prev_s_list: list,
        styletts2_params_list: list[StyleTTS2Params],
        batch_size: int,
        input_lengths: list[int],
    ) -> tuple[torch.Tensor, list, list[int], torch.Tensor | None]:
        with timed_operation("model_forward"):
            torch.compiler.cudagraph_mark_step_begin()
            
            use_diffusion_list = [p.use_diffusion for p in styletts2_params_list]
            use_diffusion_any = any(use_diffusion_list)

            with timed_operation("text_encoder"):
                t_en = self._model.text_encoder(tokens_tensor, input_lengths_tensor, text_mask)
            
            with timed_operation("bert"):
                bert_dur = self._model.bert(tokens_tensor, attention_mask=(~text_mask).int())
            
            with timed_operation("bert_encoder"):
                d_en = self._model.bert_encoder(bert_dur).transpose(-1, -2)
            
            s_pred = torch.zeros(batch_size, 256, device=self._device)
            
            if use_diffusion_any:
                with timed_operation("diffusion_sampling"):
                    diffusion_indices = [i for i in range(batch_size) if use_diffusion_list[i]]
                    non_diffusion_indices = [i for i in range(batch_size) if not use_diffusion_list[i]]
                    
                    embedding_scales = [styletts2_params_list[i].embedding_scale for i in diffusion_indices]
                    diffusion_steps_list = [styletts2_params_list[i].diffusion_steps for i in diffusion_indices]
                    
                    param_groups: dict[tuple[float, int], list[int]] = {}
                    for idx, i in enumerate(diffusion_indices):
                        key = (embedding_scales[idx], diffusion_steps_list[idx])
                        if key not in param_groups:
                            param_groups[key] = []
                        param_groups[key].append((idx, i))
                    
                    for (embedding_scale, diffusion_steps), group_items in param_groups.items():
                        group_indices_in_diffusion = [item[0] for item in group_items]
                        group_indices_in_batch = [item[1] for item in group_items]
                        group_batch_size = len(group_items)
                        
                        noise = torch.randn((group_batch_size, 256)).unsqueeze(1).to(self._device)
                        bert_dur_group = bert_dur[group_indices_in_batch]
                        ref_s_group = ref_s_batch[group_indices_in_batch]

                        if self._trt.diffusion_enabled:
                            s_pred_group = self._trt.run_diffusion_sampler(
                                noise,
                                bert_dur_group,
                                ref_s_group,
                                embedding_scale=embedding_scale,
                                diffusion_steps=diffusion_steps,
                            ).squeeze(1).clone()
                        else:
                            noise_padded, _ = self._pad_to_batch_size(noise, group_batch_size)
                            bert_dur_padded, _ = self._pad_to_batch_size(bert_dur_group, group_batch_size)
                            ref_s_batch_padded, _ = self._pad_to_batch_size(ref_s_group, group_batch_size)

                            s_pred_group_padded = self._sampler(
                                noise=noise_padded,
                                embedding=bert_dur_padded,
                                embedding_scale=embedding_scale,
                                features=ref_s_batch_padded,
                                num_steps=diffusion_steps
                            ).squeeze(1).clone()

                            s_pred_group = s_pred_group_padded[:group_batch_size]
                        style_norm = torch.linalg.vector_norm(s_pred_group.float(), dim=1)
                        use_style = torch.isfinite(s_pred_group).all(dim=1) & (style_norm <= 10.0)
                        s_pred_group = torch.where(use_style[:, None], s_pred_group, ref_s_group.to(s_pred_group.dtype))
                        for group_idx, batch_idx in enumerate(group_indices_in_batch):
                            s_pred[batch_idx] = s_pred_group[group_idx]
                    
                    if non_diffusion_indices:
                        for i in non_diffusion_indices:
                            s_pred[i] = ref_s_batch[i]
                
                if any(ps is not None for ps in prev_s_list):
                    for i in range(batch_size):
                        if use_diffusion_list[i]:
                            t = styletts2_params_list[i].style_interpolation_factor
                            if prev_s_list[i] is not None:
                                ps = prev_s_list[i].to(self._device).squeeze()
                                s_pred[i] = t * ps + (1 - t) * s_pred[i]
                
                s = s_pred[:, 128:]
                ref = s_pred[:, :128]
                
                for i in range(batch_size):
                    alpha = styletts2_params_list[i].alpha
                    beta = styletts2_params_list[i].beta
                    ref[i] = alpha * ref[i] + (1 - alpha) * ref_s_batch[i, :128]
                    s[i] = beta * s[i] + (1 - beta) * ref_s_batch[i, 128:]
            else:
                s = ref_s_batch[:, 128:]
                ref = ref_s_batch[:, :128]
            
            with timed_operation("predictor"):
                d = self._model.predictor.text_encoder(d_en, s, input_lengths_tensor, text_mask)
                
                x, _ = self._model.predictor.lstm(d)
                
                duration = self._model.predictor.duration_proj(x)
                duration = torch.sigmoid(duration).sum(axis=-1)
                pred_dur = torch.round(duration).clamp(min=1)
                pred_dur = pred_dur * (~text_mask).float()
                
                speed_tensor = torch.tensor([styletts2_params_list[i].speed for i in range(batch_size)], device=self._device).unsqueeze(1)
                pred_dur = pred_dur * speed_tensor
                pred_dur = pred_dur.clamp(min=1)

                for b in range(batch_size):
                    for token_idx in range(input_lengths[b] - 1, 0, -1):
                        symbol = self._phonemizer.index_to_symbol.get(int(tokens_tensor[b, token_idx].item()), "")
                        if symbol.isalpha():
                            break
                        pred_dur[b, token_idx] = 1
                
            decoder_type = getattr(self._model_config.decoder, 'type', None) if hasattr(self._model_config, 'decoder') else None
            
            with timed_operation("alignment_computation"):
                all_pred_aln_trg = []
                all_total_mel_frames = []
                all_actual_lengths = []
                
                for b in range(batch_size):
                    batch_pred_dur = pred_dur[b]
                    batch_input_length = input_lengths[b]
                    batch_pred_dur_actual = batch_pred_dur[:batch_input_length]
                    
                    total_mel_frames = int(batch_pred_dur_actual.sum().item())
                    all_total_mel_frames.append(total_mel_frames)
                    all_actual_lengths.append(total_mel_frames)
                    
                    if total_mel_frames == 0:
                        pred_aln_trg = torch.zeros(batch_input_length, 0, device=self._device)
                    else:
                        pred_aln_trg = torch.zeros(batch_input_length, total_mel_frames, device=self._device)
                        
                        dur_ints = batch_pred_dur_actual.int()
                        cumsum_dur = torch.cumsum(dur_ints, dim=0)
                        frame_indices = torch.arange(total_mel_frames, device=self._device).unsqueeze(0)
                        token_indices = torch.arange(batch_input_length, device=self._device).unsqueeze(1)
                        
                        start_frames = torch.cat([torch.zeros(1, device=self._device, dtype=torch.long), cumsum_dur[:-1]])
                        end_frames = cumsum_dur
                        
                        mask = (frame_indices >= start_frames.unsqueeze(1)) & (frame_indices < end_frames.unsqueeze(1))
                        pred_aln_trg[mask] = 1.0
                    
                    all_pred_aln_trg.append(pred_aln_trg)
                
                max_mel_frames = max(all_actual_lengths) if all_actual_lengths else 0
                max_input_length = max(input_lengths)
                
                batch_pred_aln_trg = torch.zeros(batch_size, max_input_length, max_mel_frames, device=self._device)
                for b in range(batch_size):
                    aln = all_pred_aln_trg[b]
                    if aln.shape[1] > 0:
                        batch_pred_aln_trg[b, :aln.shape[0], :aln.shape[1]] = aln
                
                all_pred_aln_trg = [aln.cpu().numpy() for aln in all_pred_aln_trg]  
            
            with timed_operation("F0Ntrain"):
                en = torch.bmm(d.transpose(-1, -2), batch_pred_aln_trg)
                if decoder_type == "hifigan":
                    asr_new = torch.zeros_like(en)
                    asr_new[:, :, 0] = en[:, :, 0]
                    asr_new[:, :, 1:] = en[:, :, 0:-1]
                    en = asr_new
                
                F0_pred, N_pred = self._model.predictor.F0Ntrain(en, s)
            
            with timed_operation("prepare_decoder_inputs"):
                asr = torch.bmm(t_en, batch_pred_aln_trg)
                if decoder_type == "hifigan":
                    asr_new = torch.zeros_like(asr)
                    asr_new[:, :, 0] = asr[:, :, 0]
                    asr_new[:, :, 1:] = asr[:, :, 0:-1]
                    asr = asr_new
                
                if self._trt.decoder_enabled:
                    decoder_inputs = self._trt.prepare_decoder_inputs(
                        asr,
                        F0_pred,
                        N_pred,
                        ref,
                        batch_size=batch_size,
                    )
                    asr_padded = decoder_inputs["asr"]
                    F0_pred_padded = decoder_inputs["f0"]
                    N_pred_padded = decoder_inputs["noise"]
                    ref_padded = decoder_inputs["style"]
                else:
                    asr_seq_padded = self._pad_to_multiple(asr, 128, dim=-1)
                    F0_pred_seq_padded = self._pad_to_multiple(F0_pred, 256, dim=-1)
                    N_pred_seq_padded = self._pad_to_multiple(N_pred, 256, dim=-1)

                    asr_padded, _ = self._pad_to_batch_size(asr_seq_padded, batch_size)
                    F0_pred_padded, _ = self._pad_to_batch_size(F0_pred_seq_padded, batch_size)
                    N_pred_padded, _ = self._pad_to_batch_size(N_pred_seq_padded, batch_size)
                    ref_padded, _ = self._pad_to_batch_size(ref, batch_size)
            
            with timed_operation("decoder"):
                if self._trt.decoder_enabled:
                    har = self._model.decoder.generator._preprocess_f0(F0_pred_padded)
                    out_padded = self._trt.run_decoder(
                        {
                            "asr": asr_padded,
                            "f0": F0_pred_padded,
                            "noise": N_pred_padded,
                            "style": ref_padded,
                            "har": har,
                        }
                    )
                else:
                    out_padded = self._model.decoder(asr_padded, F0_pred_padded, N_pred_padded, ref_padded)
            
            out = out_padded[:batch_size]
            
            s_pred_for_results = s_pred if use_diffusion_any else None
            
            return out, all_pred_aln_trg, all_actual_lengths, s_pred_for_results
    
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
        if getattr(self._model_config.decoder, "type", None) == "istftnet":
            decoder_hop_length = int(np.prod(self._model_config.decoder.upsample_rates) * self._model_config.decoder.gen_istft_hop_size * 2)
        max_mel_frames = max(all_actual_lengths) if all_actual_lengths else 0
        out_cpu = out[:batch_size].detach().cpu().numpy()
        ref_s_batch_cpu = ref_s_batch[:batch_size].detach().cpu().numpy()
        s_pred_cpu = s_pred[:batch_size].detach().cpu().numpy() if s_pred is not None else None
        
        for b in range(batch_size):
            use_diffusion = styletts2_params_list[b].use_diffusion
            actual_mel_frames = all_actual_lengths[b]
            audio = out_cpu[b]
            
            if audio.ndim > 1:
                audio = audio.squeeze()
            
            max_audio_length = len(audio)
            expected_audio_length = int(actual_mel_frames * decoder_hop_length)
            if 0 < expected_audio_length < len(audio):
                end_margin_samples = int(self._end_trim_margin_ms * self._sample_rate / 1000)
                audio = audio[: min(len(audio), expected_audio_length + end_margin_samples)]

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

            audio, final_alignments = _trim_leading_silence_and_shift_alignments(
                audio,
                sample_rate,
                final_alignments,
            )
            audio = np.concatenate([audio, np.zeros(int(0.15 * sample_rate), dtype=audio.dtype)])
            
            updated_style_vector = s_pred_cpu[b] if use_diffusion and s_pred_cpu is not None else ref_s_batch_cpu[b]
            
            if b < len(contexts) and contexts[b] is not None:
                contexts[b]["updated_voice"] = updated_style_vector.copy()
            
            metadata = {
                "text": original_texts[b],
                "sample_rate": sample_rate,
                "style_vector": updated_style_vector,
                "alignment_type": alignment_type.value,
                "word_alignments": final_alignments,
            }
            
            results.append(IntermediateRepresentation(
                data=audio,
                sample_rate=sample_rate,
                metadata=metadata,
            ))
        
        return results

    def _text_token_count(self, text: str, styletts2_params: StyleTTS2Params) -> int:
        if self._phonemizer is None:
            raise RuntimeError("Phonemizer not initialized")
        if styletts2_params.phonemized:
            tokens = self._phonemizer.tokenize(text)
        else:
            processed_text, _ = self._phonemizer.process_text_with_original_spans(text)
            tokens = self._phonemizer.tokenize(processed_text)
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

    def _merge_window_results(self, results: list[IntermediateRepresentation], original_text: str) -> IntermediateRepresentation:
        if len(results) == 1:
            return results[0]
        sample_rate = results[0].sample_rate
        audio = np.concatenate([np.asarray(result.data) for result in results], axis=-1)
        metadata = dict(results[-1].metadata)
        metadata["text"] = original_text
        metadata["window_count"] = len(results)
        return IntermediateRepresentation(data=audio, sample_rate=sample_rate, metadata=metadata)

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
                self._prepare_voice_contexts(contexts, batch_size)

            styletts2_params_list = []
            for i in range(batch_size):
                valid_params = {f.name: params[i][f.name] for f in fields(StyleTTS2Params) if f.name in params[i]}
                styletts2_params_list.append(StyleTTS2Params(**valid_params))

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
    
    
