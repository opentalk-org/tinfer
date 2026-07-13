from __future__ import annotations

from typing import Any

import numpy as np
import torch

class VoiceContextMixin:
    def _validate_generation_ready(self) -> None:
        if not self._loaded:
            raise RuntimeError("Model not loaded. Call load() first.")
        
        if self._model is None:
            raise RuntimeError("Model not initialized")
    
    def _prepare_voice_contexts(self, texts: list[str], contexts: list[dict[str, Any] | None], batch_size: int) -> None:
        for i in range(batch_size):
            if i >= len(contexts):
                contexts.append({})
            elif contexts[i] is None:
                contexts[i] = {}
            context = contexts[i]
            
            if context.get("reset_voice", False):
                context.pop("previous_style_vector", None)
                context["reset_voice"] = False
            
            voice_id = context.get("voice_id")
            if voice_id == "auto":
                voice_entry = self._voice_cache.pick_auto(texts[i])
                if context.get("base_voice_id") != voice_entry.voice_id:
                    base_voice_np = voice_entry.tensor.cpu().numpy() if isinstance(voice_entry.tensor, torch.Tensor) else voice_entry.tensor
                    context["base_voice"] = base_voice_np.copy() if isinstance(base_voice_np, np.ndarray) else base_voice_np
                    context["base_voice_id"] = voice_entry.voice_id
                    context["auto_voice_id"] = voice_entry.voice_id
                    context["auto_voice_source"] = voice_entry.source_file
                continue

            if voice_id is not None and context.get("base_voice_id") != voice_id:
                if not self.has_voice(voice_id):
                    raise ValueError(f"Voice ID '{voice_id}' not found in cache. Available voices: {self.list_voices()}")
                
                base_voice_tensor = self.get_voice(voice_id)
                base_voice_np = base_voice_tensor.cpu().numpy() if isinstance(base_voice_tensor, torch.Tensor) else base_voice_tensor
                context["base_voice"] = base_voice_np.copy() if isinstance(base_voice_np, np.ndarray) else base_voice_np
                context["base_voice_id"] = voice_id
    
    def _prepare_voice_tensors(self, contexts: list[dict[str, Any] | None], batch_size: int) -> torch.Tensor:
        ref_s_list = []
        for i in range(batch_size):
            context = contexts[i] if i < len(contexts) and contexts[i] is not None else {}
            
            if "base_voice" in context:
                ref_s = context["base_voice"]
            else:
                raise ValueError(f"Voice conditioning required for StyleTTS2 (request {i}). Provide voice_id or base_voice in context.")
            
            ref_s = torch.from_numpy(ref_s).to(self._device)
            if ref_s.dim() == 1:
                ref_s = ref_s.unsqueeze(0)
            ref_s_list.append(ref_s)
        
        ref_s_batch = torch.cat(ref_s_list, dim=0)
        return ref_s_batch
    
    def _prepare_previous_style_vectors(self, contexts: list[dict[str, Any] | None], batch_size: int) -> list:
        prev_s_list = []
        for i in range(batch_size):
            context = contexts[i] if i < len(contexts) and contexts[i] is not None else {}
            if "previous_style_vector" in context:
                prev_s = context["previous_style_vector"]
                if isinstance(prev_s, np.ndarray):
                    prev_s = torch.from_numpy(prev_s)
                prev_s = prev_s.to(self._device)
                if prev_s.dim() == 1:
                    prev_s = prev_s.unsqueeze(0)
                prev_s_list.append(prev_s)
            else:
                prev_s_list.append(None)
        
        return prev_s_list
    
