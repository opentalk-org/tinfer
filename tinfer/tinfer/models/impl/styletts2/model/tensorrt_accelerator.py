from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any, Mapping

import torch

from tinfer.models.impl.styletts2.model.tensorrt_config import get_tensorrt_model_config


def pad_or_clip_to_size(tensor: torch.Tensor, target_size: int, dim: int = -1) -> torch.Tensor:
    if dim < 0:
        dim = len(tensor.shape) + dim
    current_size = tensor.shape[dim]
    if current_size == target_size:
        return tensor
    if current_size > target_size:
        slices = [slice(None)] * tensor.dim()
        slices[dim] = slice(0, target_size)
        return tensor[tuple(slices)]

    padding_size = target_size - current_size
    pad_shape = list(tensor.shape)
    pad_shape[dim] = padding_size
    last_slice = tensor.select(dim, current_size - 1).unsqueeze(dim)
    padding = last_slice.expand(*pad_shape)
    return torch.cat([tensor, padding], dim=dim)


class StyleTTS2TensorRTAccelerator:
    def __init__(self) -> None:
        self._runtime: Any | None = None
        self._config: Mapping[str, Any] | None = None
        self._decoder_enabled = False
        self._engine_dir: str | None = None
        self._max_batch = 16
        self._max_asr_frames = 1024
        self._decoder_dynamic_runner: Any | None = None
        self._diffusion_enabled = False
        self._diffusion_engine_dir: str | None = None
        self._max_diffusion_batch = 16
        self._min_diffusion_tokens = 16
        self._max_diffusion_tokens = 512
        self._diffusion_dynamic_runners: dict[int, Any] = {}

    @classmethod
    def from_runtime_config(
        cls,
        runtime_config: Mapping[str, Any],
        model_dir: str | Path,
        model: Any | None,
        model_config: Any | None,
    ) -> "StyleTTS2TensorRTAccelerator":
        accelerator = cls()
        trt_config = get_tensorrt_model_config(runtime_config, model_dir)
        if trt_config is None:
            return accelerator

        accelerator._runtime = importlib.import_module(
            "tinfer.models.impl.styletts2.model.modules.tensorrt_runtime"
        )
        accelerator._config = trt_config
        accelerator._configure_decoder(trt_config, model, model_config)
        accelerator._configure_diffusion(trt_config, model)
        return accelerator

    @property
    def decoder_enabled(self) -> bool:
        return self._decoder_enabled

    @property
    def diffusion_enabled(self) -> bool:
        return self._diffusion_enabled

    def _configure_decoder(self, trt_config: Mapping[str, Any], model: Any | None, model_config: Any | None) -> None:
        self._decoder_enabled = bool(
            "decoder" in trt_config.get("components", [])
            and isinstance(trt_config.get("decoder"), dict)
        )
        if not self._decoder_enabled:
            return
        if model is None or model_config is None:
            return
        decoder_type = getattr(model_config.decoder, "type", None) if hasattr(model_config, "decoder") else None
        if decoder_type != "istftnet":
            raise RuntimeError(f"TensorRT decoder mode supports istftnet decoder only, got {decoder_type!r}")
        decoder_config = trt_config["decoder"]
        self._engine_dir = str(trt_config["engine_dir"])
        self._max_batch = int(decoder_config["max_batch"])
        self._max_asr_frames = int(decoder_config["max_asr_frames"])
        self._remove_decoder_weight_norm(model.decoder)
        self._preload_decoder_runner()

    def _configure_diffusion(self, trt_config: Mapping[str, Any], model: Any | None) -> None:
        self._diffusion_enabled = bool(
            "diffusion" in trt_config.get("components", [])
            and isinstance(trt_config.get("diffusion"), dict)
        )
        if not self._diffusion_enabled:
            return
        if model is None:
            return
        diffusion_config = trt_config["diffusion"]
        self._diffusion_engine_dir = str(trt_config["engine_dir"])
        self._max_diffusion_batch = int(diffusion_config["max_batch"])
        self._min_diffusion_tokens = int(diffusion_config["min_tokens"])
        self._max_diffusion_tokens = int(diffusion_config["max_tokens"])
        self._preload_diffusion_runners()

    def _preload_decoder_runner(self) -> None:
        if self._engine_dir is None or self._runtime is None:
            return
        engine_dir = Path(self._engine_dir)
        dynamic_path = engine_dir / self._runtime.DynamicDecoderTRTEngineSpec().file_name
        if dynamic_path.exists():
            self._decoder_dynamic_runner = self._runtime.get_tensorrt_decoder_runner(engine_dir)

    def _preload_diffusion_runners(self) -> None:
        if self._diffusion_engine_dir is None or self._runtime is None:
            return
        engine_dir = Path(self._diffusion_engine_dir)
        diffusion_config = self._config.get("diffusion", {}) if self._config else {}
        for engine in diffusion_config.get("engines", []):
            spec = self._runtime.DynamicDiffusionTRTEngineSpec(num_steps=int(engine["steps"]))
            if spec.num_steps not in self._diffusion_dynamic_runners:
                self._diffusion_dynamic_runners[spec.num_steps] = self._runtime.get_tensorrt_diffusion_runner(
                    engine_dir,
                    num_steps=spec.num_steps,
                )

    def _remove_decoder_weight_norm(self, decoder: Any) -> None:
        for module in decoder.modules():
            try:
                torch.nn.utils.remove_weight_norm(module)
            except ValueError:
                pass

    def prepare_decoder_inputs(
        self,
        asr: torch.Tensor,
        f0: torch.Tensor,
        noise: torch.Tensor,
        style: torch.Tensor,
        *,
        batch_size: int,
    ) -> dict[str, torch.Tensor]:
        if batch_size > self._max_batch:
            raise RuntimeError(f"TensorRT decoder batch {batch_size} exceeds max {self._max_batch}")
        target_asr_frames = min(max(asr.shape[-1], 128), self._max_asr_frames)
        target_f0_frames = target_asr_frames * 2
        asr_padded = pad_or_clip_to_size(asr, target_asr_frames, dim=-1)
        f0_padded = pad_or_clip_to_size(f0, target_f0_frames, dim=-1)
        noise_padded = pad_or_clip_to_size(noise, target_f0_frames, dim=-1)
        if asr.shape[-1] < target_asr_frames:
            asr_padded[:, :, asr.shape[-1]:] = 0
            f0_padded[:, f0.shape[-1]:] = 0
            noise_padded[:, noise.shape[-1]:] = 0
        return {
            "asr": asr_padded,
            "f0": f0_padded,
            "noise": noise_padded,
            "style": style,
        }

    def run_decoder(self, inputs: Mapping[str, torch.Tensor]) -> torch.Tensor:
        return self._get_decoder_runner().run(inputs)["audio"]

    def run_diffusion_sampler(
        self,
        noise: torch.Tensor,
        embedding: torch.Tensor,
        features: torch.Tensor,
        *,
        embedding_scale: float,
        diffusion_steps: int,
    ) -> torch.Tensor:
        if embedding_scale != 1.0:
            raise RuntimeError("TensorRT diffusion currently supports embedding_scale=1.0 only")

        if noise.shape[0] > self._max_diffusion_batch:
            raise RuntimeError(
                f"TensorRT diffusion batch {noise.shape[0]} exceeds max {self._max_diffusion_batch}"
            )
        if self._runtime is None:
            raise RuntimeError("TensorRT runtime is not configured")

        target_tokens = min(max(embedding.shape[1], self._min_diffusion_tokens), self._max_diffusion_tokens)
        bucket = self._runtime.DiffusionShapeBucket(
            batch_size=noise.shape[0],
            embedding_tokens=target_tokens,
            num_steps=diffusion_steps,
        )
        embedding_padded = pad_or_clip_to_size(embedding, target_tokens, dim=1)
        step_noise = torch.randn(
            bucket.batch_size,
            diffusion_steps - 1,
            1,
            noise.shape[-1],
            device=noise.device,
            dtype=noise.dtype,
        )

        out = self._get_diffusion_runner(bucket).run(
            {
                "noise": noise,
                "step_noise": step_noise,
                "embedding": embedding_padded,
                "features": features,
            }
        )["style"]
        return out[: noise.shape[0]].to(dtype=noise.dtype)

    def _get_decoder_runner(self) -> Any:
        if self._decoder_dynamic_runner is not None:
            return self._decoder_dynamic_runner
        raise RuntimeError("Dynamic TensorRT decoder engine is not configured")

    def _get_diffusion_runner(self, bucket: Any) -> Any:
        dynamic_runner = self._diffusion_dynamic_runners.get(bucket.num_steps)
        if dynamic_runner is not None:
            return dynamic_runner
        raise RuntimeError(f"Dynamic TensorRT diffusion engine for {bucket.num_steps} steps is not configured")
