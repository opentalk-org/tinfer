from pathlib import Path
import torch


class VoiceCache:
    def __init__(self):
        self._cache: dict[str, torch.Tensor] = {}

    def get(self, voice_id: str) -> torch.Tensor:
        return self._cache[voice_id]

    def put(self, voice_id: str, voice_vector: torch.Tensor):
        self._cache[voice_id] = voice_vector

    def load(self, voices_path: str):
        voices_path = Path(voices_path)
        if not voices_path.exists():
            raise ValueError(f"Voices path does not exist: {voices_path}")
        
        for voice_path in voices_path.glob("*.pth"):
            voice_id = voice_path.stem
            voice_data = torch.load(voice_path, map_location='cpu', weights_only=True)
            
            voice_vector = voice_data['voice_vector']
            
            if not isinstance(voice_vector, torch.Tensor):
                raise ValueError(f"Extracted voice data from {voice_path} is not a torch.Tensor, got {type(voice_vector)}")
            
            self._cache[voice_id] = voice_vector

    def clear(self):
        self._cache.clear()
    
    def __len__(self) -> int:
        return len(self._cache)
