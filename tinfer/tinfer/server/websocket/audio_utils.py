from __future__ import annotations

from tinfer.utils.audio_encoder import (
    AudioFormat,
    AudioEncoder,
    DefaultAudioEncoder,
    parse_output_format,
    encode_audio_to_base64,
    get_sample_rate,
)

WebSocketAudioEncoder = DefaultAudioEncoder
