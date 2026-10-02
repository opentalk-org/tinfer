import base64
import json
import wave
from dataclasses import asdict, dataclass
from pathlib import Path
from urllib.parse import quote
from urllib.request import Request, urlopen

from config import model_id, sample_rate, server_url, voice_id


@dataclass
class VoiceSettings:
    speed: float = 1.0
    alpha: float = 0.3
    beta: float = 0.7
    stability: float = 0.7


@dataclass
class Speech:
    text: str
    model_id: str
    voice_settings: VoiceSettings


@dataclass
class Timing:
    characters: list[str]
    character_start_times_seconds: list[float]
    character_end_times_seconds: list[float]


@dataclass
class TimedSpeech:
    audio: bytes
    alignment: Timing


def speech_request(text: str, model: str, voice: str, suffix: str, settings: VoiceSettings) -> Request:
    body = Speech(text, model, settings)
    url = f"{server_url}/v1/text-to-speech/{quote(voice, safe='')}{suffix}?output_format=pcm_24000"
    return Request(url, data=json.dumps(asdict(body)).encode(), headers={"Content-Type": "application/json"})


def synthesize(text: str, model: str = model_id, voice: str = voice_id) -> bytes:
    request = speech_request(text, model, voice, "", VoiceSettings())
    with urlopen(request, timeout=180) as response:
        return response.read()


def synthesize_timed(text: str) -> TimedSpeech:
    request = speech_request(text, model_id, voice_id, "/with-timestamps", VoiceSettings())
    with urlopen(request, timeout=180) as response:
        value = json.load(response)
    return TimedSpeech(base64.b64decode(value["audio_base64"]), Timing(**value["alignment"]))


def save_audio(audio: bytes, path: Path) -> None:
    assert audio and len(audio) % 2 == 0, "expected nonempty PCM16 audio"
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(sample_rate)
        output.writeframes(audio)
    print(f"Saved {len(audio) / (2 * sample_rate):.3f}s audio to {path}")
