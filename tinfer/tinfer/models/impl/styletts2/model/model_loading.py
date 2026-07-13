from __future__ import annotations

import importlib
from dataclasses import asdict
from pathlib import Path
from typing import Any
import warnings

import librosa
import torch
from munch import Munch

from tinfer.models.impl.styletts2.model.modules.diffusion.sampler import ADPM2Sampler, DiffusionSampler, KarrasSchedule
from tinfer.models.impl.styletts2.model.modules.load_utils import load_model, load_model_from_state
from tinfer.models.impl.styletts2.model.tensorrt_accelerator import (
    StyleTTS2TensorRTAccelerator,
)
from tinfer.support.observability import get_logger
from .text_config import TextConfig

log = get_logger(__name__)

class ModelLoadingMixin:
    def _initialize_from_config(
        self,
        config: dict[str, Any],
        text_config: TextConfig,
        device: str,
    ) -> None:
        self._config = config
        self._text_config = text_config
        self._default_language = text_config.default_language
        self._sample_rate = config.get("sample_rate", 24000)
        self.max_batch_size = config.get("max_batch_size", 10)
        
        load_style_encoder = config.get("load_style_encoder", True)

        self._model = Munch({key: self._model[key].to(self._device) for key in self._model})
        _ = [self._model[key].eval() for key in self._model]

        self._phonemizer = self._get_phonemizer(self._default_language)
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
        log.info("decoder_compile_started")
        self._model['decoder'].generator._forward_compiled = torch.compile(
            self._model['decoder'].generator._forward_compiled,
            mode='reduce-overhead',
            dynamic=True
        )
        log.info("decoder_compile_finished")

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
        text_config = TextConfig.from_dict(
            model_saved['text_config'],
            self._model_config.n_token,
        )
        self._initialize_from_config(config, text_config, device)
        
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
        assert self._text_config is not None, "loaded model requires text config"
        state_dict['text_config'] = self._text_config.to_dict()
        
        return state_dict

    def load_from_state(self, state: dict[str, Any], device: str = "cuda") -> None:
        config = state.get('runtime_config', {})
        load_style_encoder = config.get("load_style_encoder", True)
        
        self._model, self._model_config = load_model_from_state(state, load_style_encoder)
        text_config = TextConfig.from_dict(state['text_config'], self._model_config.n_token)
        self._initialize_from_config(config, text_config, device)
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
    
