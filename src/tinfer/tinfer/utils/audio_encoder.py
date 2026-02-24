from __future__ import annotations
from abc import ABC, abstractmethod
from enum import Enum
from typing import Any
import base64
import io
import subprocess
import audioop

import numpy as np
import librosa
from pydub import AudioSegment


class AudioFormat(Enum):
    """
    Supported audio output formats.
    
    Values:
        MP3_22050_32: MP3 at 22050 Hz, 32 kbps
        MP3_44100_32: MP3 at 44100 Hz, 32 kbps
        MP3_44100_64: MP3 at 44100 Hz, 64 kbps
        MP3_44100_96: MP3 at 44100 Hz, 96 kbps
        MP3_44100_128: MP3 at 44100 Hz, 128 kbps
        MP3_44100_192: MP3 at 44100 Hz, 192 kbps
        PCM_8000: PCM at 8000 Hz
        PCM_16000: PCM at 16000 Hz
        PCM_22050: PCM at 22050 Hz
        PCM_24000: PCM at 24000 Hz
        PCM_44100: PCM at 44100 Hz
        ULAW_8000: μ-law at 8000 Hz
        ALAW_8000: A-law at 8000 Hz
        OPUS_48000_32: Opus at 48000 Hz, 32 kbps
        OPUS_48000_64: Opus at 48000 Hz, 64 kbps
        OPUS_48000_96: Opus at 48000 Hz, 96 kbps
        OPUS_48000_128: Opus at 48000 Hz, 128 kbps
        OPUS_48000_192: Opus at 48000 Hz, 192 kbps
    """

    MP3_22050_32 = "mp3_22050_32"
    MP3_44100_32 = "mp3_44100_32"
    MP3_44100_64 = "mp3_44100_64"
    MP3_44100_96 = "mp3_44100_96"
    MP3_44100_128 = "mp3_44100_128"
    MP3_44100_192 = "mp3_44100_192"
    PCM_8000 = "pcm_8000"
    PCM_16000 = "pcm_16000"
    PCM_22050 = "pcm_22050"
    PCM_24000 = "pcm_24000"
    PCM_44100 = "pcm_44100"
    ULAW_8000 = "ulaw_8000"
    ALAW_8000 = "alaw_8000"
    OPUS_48000_32 = "opus_48000_32"
    OPUS_48000_64 = "opus_48000_64"
    OPUS_48000_96 = "opus_48000_96"
    OPUS_48000_128 = "opus_48000_128"
    OPUS_48000_192 = "opus_48000_192"

# TODO: implement fast audio encoder
class AudioEncoder(ABC):
    @abstractmethod
    def encode(
        self, audio: np.ndarray, sample_rate: int, format: AudioFormat
    ) -> bytes:
        pass

    @abstractmethod
    def encode_base64(
        self, audio: np.ndarray, sample_rate: int, format: AudioFormat
    ) -> str:
        pass

    @abstractmethod
    def get_sample_rate(self, format: AudioFormat) -> int:
        pass


def parse_output_format(format_str: str) -> AudioFormat:
    format_str = format_str.lower().strip()
    
    try:
        return AudioFormat(format_str)
    except ValueError:
        if format_str.startswith("mp3"):
            if "22050" in format_str:
                if "32" in format_str:
                    return AudioFormat.MP3_22050_32
            elif "44100" in format_str:
                if "32" in format_str:
                    return AudioFormat.MP3_44100_32
                elif "64" in format_str:
                    return AudioFormat.MP3_44100_64
                elif "96" in format_str:
                    return AudioFormat.MP3_44100_96
                elif "128" in format_str:
                    return AudioFormat.MP3_44100_128
                elif "192" in format_str:
                    return AudioFormat.MP3_44100_192
        elif format_str.startswith("pcm"):
            if "8000" in format_str:
                return AudioFormat.PCM_8000
            elif "16000" in format_str:
                return AudioFormat.PCM_16000
            elif "22050" in format_str:
                return AudioFormat.PCM_22050
            elif "24000" in format_str:
                return AudioFormat.PCM_24000
            elif "44100" in format_str:
                return AudioFormat.PCM_44100
        elif format_str.startswith("ulaw"):
            return AudioFormat.ULAW_8000
        elif format_str.startswith("alaw"):
            return AudioFormat.ALAW_8000
        elif format_str.startswith("opus"):
            if "48000" in format_str:
                if "32" in format_str:
                    return AudioFormat.OPUS_48000_32
                elif "64" in format_str:
                    return AudioFormat.OPUS_48000_64
                elif "96" in format_str:
                    return AudioFormat.OPUS_48000_96
                elif "128" in format_str:
                    return AudioFormat.OPUS_48000_128
                elif "192" in format_str:
                    return AudioFormat.OPUS_48000_192
        
        return AudioFormat.MP3_44100_32


class DefaultAudioEncoder(AudioEncoder):
    def get_sample_rate(self, format: AudioFormat) -> int:
        format_str = format.value
        if "8000" in format_str:
            return 8000
        elif "16000" in format_str:
            return 16000
        elif "22050" in format_str:
            return 22050
        elif "24000" in format_str:
            return 24000
        elif "44100" in format_str:
            return 44100
        elif "48000" in format_str:
            return 48000
        return 24000

    def encode(
        self, audio: np.ndarray, sample_rate: int, format: AudioFormat
    ) -> bytes:
        format_str = format.value
        
        target_rate = self.get_sample_rate(format)
        if sample_rate != target_rate:
            audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=target_rate)
            sample_rate = target_rate
        
        if format_str.startswith("pcm"):
            audio_int16 = (audio * 32767.0).astype(np.int16)
            return audio_int16.tobytes()
        
        elif format_str.startswith("ulaw"):
            audio_int16 = (audio * 32767.0).astype(np.int16)
            return audioop.lin2ulaw(audio_int16.tobytes(), 2)
        
        elif format_str.startswith("alaw"):
            audio_int16 = (audio * 32767.0).astype(np.int16)
            return audioop.lin2alaw(audio_int16.tobytes(), 2)
        
        elif format_str.startswith("mp3") or format_str.startswith("opus"):
            return self._encode_with_ffmpeg(audio, sample_rate, format)
        
        else:
            audio_int16 = (audio * 32767.0).astype(np.int16)
            return audio_int16.tobytes()

    def encode_base64(
        self, audio: np.ndarray, sample_rate: int, format: AudioFormat
    ) -> str:
        audio_bytes = self.encode(audio, sample_rate, format)
        return base64.b64encode(audio_bytes).decode("utf-8")

    def _encode_with_ffmpeg(
        self,
        audio: np.ndarray,
        sample_rate: int,
        audio_format: AudioFormat,
    ) -> bytes:
        format_str = audio_format.value
        
        if format_str.startswith("mp3"):
            bitrate = "32k"
            if "64" in format_str:
                bitrate = "64k"
            elif "96" in format_str:
                bitrate = "96k"
            elif "128" in format_str:
                bitrate = "128k"
            elif "192" in format_str:
                bitrate = "192k"
            
            audio_int16 = (audio * 32767.0).astype(np.int16)
            
            try:
                audio_segment = AudioSegment(
                    audio_int16.tobytes(),
                    frame_rate=sample_rate,
                    channels=1,
                    sample_width=2,
                )
                buffer = io.BytesIO()
                audio_segment.export(buffer, format="mp3", bitrate=bitrate)
                return buffer.getvalue()
            except Exception:
                try:
                    process = subprocess.Popen(
                        [
                            "ffmpeg",
                            "-f", "s16le",
                            "-ar", str(sample_rate),
                            "-ac", "1",
                            "-i", "pipe:0",
                            "-f", "mp3",
                            "-b:a", bitrate,
                            "pipe:1",
                        ],
                        stdin=subprocess.PIPE,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                    )
                    stdout, stderr = process.communicate(input=audio_int16.tobytes())
                    if process.returncode == 0:
                        return stdout
                except Exception:
                    pass
            
            audio_int16 = (audio * 32767.0).astype(np.int16)
            return audio_int16.tobytes()
        
        elif format_str.startswith("opus"):
            bitrate = "32k"
            if "64" in format_str:
                bitrate = "64k"
            elif "96" in format_str:
                bitrate = "96k"
            elif "128" in format_str:
                bitrate = "128k"
            elif "192" in format_str:
                bitrate = "192k"
            
            audio_int16 = (audio * 32767.0).astype(np.int16)
            
            try:
                process = subprocess.Popen(
                    [
                        "ffmpeg",
                        "-f", "s16le",
                        "-ar", str(sample_rate),
                        "-ac", "1",
                        "-i", "pipe:0",
                        "-f", "opus",
                        "-b:a", bitrate,
                        "pipe:1",
                    ],
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                )
                stdout, stderr = process.communicate(input=audio_int16.tobytes())
                if process.returncode == 0:
                    return stdout
            except Exception:
                pass
            
            audio_int16 = (audio * 32767.0).astype(np.int16)
            return audio_int16.tobytes()
        
        audio_int16 = (audio * 32767.0).astype(np.int16)
        return audio_int16.tobytes()


_default_encoder = DefaultAudioEncoder()


def encode_audio_to_base64(
    audio: np.ndarray,
    sample_rate: int,
    audio_format: AudioFormat,
) -> str:
    return _default_encoder.encode_base64(audio, sample_rate, audio_format)


def get_sample_rate(audio_format: AudioFormat) -> int:
    return _default_encoder.get_sample_rate(audio_format)
