import torch
import torch.nn as nn
import numpy as np
import librosa
from typing import Optional, Any

from tinfer.models.base.voice import VoiceEncoder
from munch import Munch
class StyleTTS2VoiceEncoder(VoiceEncoder):
    def __init__(self, model: dict[str, nn.Module], device: str, sample_rate: int):
        self.model = model
        self.device = device
        self.sample_rate = sample_rate

    def preprocess(self, wave: np.ndarray) -> torch.Tensor:
        wave_tensor = torch.from_numpy(wave).float()
        mel_tensor = self.to_mel(wave_tensor)
        mel_tensor = (torch.log(1e-5 + mel_tensor.unsqueeze(0)) - self.mean) / self.std
        return mel_tensor

    def to(self, device: str) -> None:
        self.device = device
        self.model = {key: self.model[key].to(device) for key in self.model}
        _ = Munch({key: self.model[key].eval() for key in self.model})
    
    def compute_style_from_waveform(
        self,
        waveform: np.ndarray,
        sample_rate: int,
        model: Optional[Any] = None
    ) -> torch.Tensor:
        if model is None:
            model = self.model
        
        if model is None:
            raise ValueError("Model must be provided either in __init__ or compute_style_from_waveform")
        
        if sample_rate != self.sample_rate:
            waveform = librosa.resample(waveform, orig_sr=sample_rate, target_sr=self.sample_rate)
        
        audio, index = librosa.effects.trim(waveform, top_db=30)
        mel_tensor = self.preprocess(audio).to(self.device)
        
        with torch.no_grad():
            ref_s = model.style_encoder(mel_tensor.unsqueeze(1))
            ref_p = model.predictor_encoder(mel_tensor.unsqueeze(1))
        
        return torch.cat([ref_s, ref_p], dim=1)

    def encode(self, audio: np.ndarray, sample_rate: int) -> torch.Tensor:
        voice_vector = self.compute_style_from_waveform(audio, sample_rate)
        return voice_vector