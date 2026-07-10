from pathlib import Path
from dataclasses import dataclass
import unicodedata

import torch


@dataclass(frozen=True)
class VoiceLength:
    words: int
    chars: int


@dataclass(frozen=True)
class VoiceEntry:
    voice_id: str
    tensor: torch.Tensor
    text: str | None = None
    length: VoiceLength | None = None
    source_file: str | None = None


class VoiceCache:
    def __init__(self):
        self._cache: dict[str, VoiceEntry] = {}

    def get(self, voice_id: str) -> torch.Tensor:
        return self._cache[voice_id].tensor

    def get_entry(self, voice_id: str) -> VoiceEntry:
        return self._cache[voice_id]

    def put(self, voice_id: str, voice_vector: torch.Tensor):
        self._cache[voice_id] = VoiceEntry(voice_id=voice_id, tensor=voice_vector)

    def load(self, voices_path: str):
        voices_path = Path(voices_path)
        if not voices_path.exists():
            raise ValueError(f"Voices path does not exist: {voices_path}")

        for voice_path in voices_path.glob("*.pth"):
            voice_id = voice_path.stem
            voice_data = torch.load(voice_path, map_location='cpu', weights_only=True)

            entry = self._entry_from_saved_voice(voice_id, voice_path, voice_data)
            self._cache[voice_id] = entry

    def pick_auto(self, text: str) -> VoiceEntry:
        last_punct = terminal_punctuation(text)
        lens = text_len(text)
        entries_with_text = [entry for entry in self._cache.values() if entry.text]
        candidates = [entry for entry in entries_with_text if terminal_punctuation(entry.text or "") == last_punct]
        if not candidates:
            candidates = entries_with_text
        if not candidates:
            raise ValueError("voice_id='auto' requires voice metadata with text")

        return sorted(candidates, key=lambda entry: voice_distance(entry, lens))[0]

    def clear(self):
        self._cache.clear()

    def __len__(self) -> int:
        return len(self._cache)

    def _entry_from_saved_voice(self, voice_id: str, voice_path: Path, voice_data: object) -> VoiceEntry:
        if isinstance(voice_data, torch.Tensor):
            return VoiceEntry(voice_id=voice_id, tensor=voice_data)

        if isinstance(voice_data, dict) and "tensor" in voice_data:
            voice_vector = voice_data["tensor"]
            if not isinstance(voice_vector, torch.Tensor):
                voice_vector = torch.tensor(voice_vector, dtype=torch.float32)
            text = voice_data["text"] if "text" in voice_data else None
            length = voice_length(voice_data["len"]) if "len" in voice_data and voice_data["len"] else None
            source_file = voice_data["file"] if "file" in voice_data else None
            return VoiceEntry(voice_id=voice_id, tensor=voice_vector, text=text, length=length, source_file=source_file)

        raise ValueError(f"Extracted voice data from {voice_path} must be a torch.Tensor or metadata dict, got {type(voice_data)}")


def normalize_text(text: str) -> str:
    return unicodedata.normalize("NFKC", text).casefold()


def terminal_punctuation(text: str) -> str:
    stripped = normalize_text(text).strip()
    for char in reversed(stripped):
        if char == "?":
            return "?"
        if char in ".!":
            return "."
        if not char.isspace():
            return "."
    return "."


def cleanup(text: str) -> str:
    normalized = normalize_text(text)
    return "".join(char for char in normalized if not char.isspace())


def text_len(text: str) -> VoiceLength:
    normalized = normalize_text(text)
    word_count = 0
    in_word = False
    for char in normalized:
        if char.isalnum():
            if not in_word:
                word_count += 1
                in_word = True
        else:
            in_word = False
    return VoiceLength(words=word_count, chars=len(cleanup(normalized)))


def voice_length(value: dict[str, int]) -> VoiceLength:
    return VoiceLength(words=int(value["words"]), chars=int(value["chars"]))


def voice_distance(entry: VoiceEntry, lens: VoiceLength) -> tuple[int, int]:
    assert entry.text is not None
    entry_len = entry.length if entry.length is not None else text_len(entry.text)
    return abs(entry_len.words - lens.words), abs(entry_len.chars - lens.chars)
