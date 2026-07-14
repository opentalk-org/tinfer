use super::mp3::Mp3Encoder;
use super::opus::OpusEncoder;
use super::{AudioEncoding, AudioError, AudioFormat, Codec, Result};

pub(crate) enum CompressedEncoder {
    Mp3(Mp3Encoder),
    Opus(OpusEncoder),
}

pub(crate) fn check_native(codec: Codec, code: i32) -> Result<i32> {
    if code < 0 { Err(AudioError::CodecFailure { codec, code }) } else { Ok(code) }
}

impl CompressedEncoder {
    pub(crate) fn new(format: AudioFormat, source_rate: u32) -> Result<Self> {
        match format.encoding {
            AudioEncoding::Mp3 => Mp3Encoder::new(format).map(Self::Mp3),
            AudioEncoding::Opus => OpusEncoder::new(format, source_rate).map(Self::Opus),
            _ => unreachable!("compressed encoder requires a compressed audio format"),
        }
    }

    pub(crate) fn push(&mut self, samples: &[f32]) -> Result<Vec<u8>> {
        match self {
            Self::Mp3(encoder) => encoder.push(samples),
            Self::Opus(encoder) => encoder.push(samples),
        }
    }

    pub(crate) fn finish(&mut self) -> Result<Vec<u8>> {
        match self {
            Self::Mp3(encoder) => encoder.finish(),
            Self::Opus(encoder) => encoder.finish(),
        }
    }
}
