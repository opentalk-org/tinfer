mod compressed;
mod format;
mod mp3;
mod ogg;
mod opus;
mod pcm;
mod resample;
mod wav;

use bytes::Bytes;
use thiserror::Error;

use compressed::CompressedEncoder;
pub use format::{AudioEncoding, AudioFormat};
use resample::StreamResampler;

pub type Result<T> = std::result::Result<T, AudioError>;

#[derive(Clone, Debug, Eq, Error, PartialEq)]
pub enum AudioError {
    #[error("unknown audio format: {0}")]
    UnknownFormat(String),
    #[error("unsupported source sample rate: {0}")]
    UnsupportedSampleRate(u32),
    #[error("audio contains a non-finite sample")]
    NonFiniteSample,
    #[error("audio encoder is already finished")]
    Finished,
    #[error("audio frame count exceeds supported size")]
    AudioTooLong,
    #[error("audio exceeds the RIFF/WAVE u32 size limit")]
    WavTooLarge,
    #[error("resampling failed: {0}")]
    Resample(String),
    #[error("{codec} codec library is unavailable: {reason}")]
    CodecUnavailable { codec: Codec, reason: String },
    #[error("{0} codec initialization failed")]
    CodecInitialization(Codec),
    #[error("{codec} codec failed with native code {code}")]
    CodecFailure { codec: Codec, code: i32 },
}

#[derive(Clone, Copy, Debug, Eq, Error, PartialEq)]
pub enum Codec {
    #[error("MP3")]
    Mp3,
    #[error("Opus")]
    Opus,
}

pub struct AudioEncoder {
    format: AudioFormat,
    resampler: StreamResampler,
    wav_pcm: Vec<u8>,
    compressed: Option<CompressedEncoder>,
    finished: bool,
}

impl AudioEncoder {
    pub fn new(format: AudioFormat, source_rate: u32) -> Result<Self> {
        if !matches!(source_rate, 8_000 | 16_000 | 22_050 | 24_000 | 32_000 | 44_100 | 48_000) {
            return Err(AudioError::UnsupportedSampleRate(source_rate));
        }
        let compressed = matches!(format.encoding, AudioEncoding::Mp3 | AudioEncoding::Opus)
            .then(|| CompressedEncoder::new(format, source_rate))
            .transpose()?;
        Ok(Self {
            format,
            resampler: StreamResampler::new(source_rate, format.sample_rate)?,
            wav_pcm: Vec::new(),
            compressed,
            finished: false,
        })
    }

    pub fn push(&mut self, samples: &[f32]) -> Result<Bytes> {
        self.ensure_active()?;
        if samples.iter().any(|sample| !sample.is_finite()) {
            return Err(AudioError::NonFiniteSample);
        }
        let resampled = self.resampler.push(samples)?;
        self.encode(&resampled)
    }

    pub fn finish(&mut self) -> Result<Bytes> {
        self.ensure_active()?;
        let tail = self.resampler.finish()?;
        let mut tail_bytes = self.encode(&tail)?.to_vec();
        self.finished = true;
        match self.format.encoding {
            AudioEncoding::WavPcm16 => Ok(Bytes::from(wav::encode(self.format.sample_rate, &self.wav_pcm)?)),
            AudioEncoding::Mp3 | AudioEncoding::Opus => {
                tail_bytes.extend(self.compressed.as_mut().expect("compressed format has codec state").finish()?);
                Ok(Bytes::from(tail_bytes))
            }
            _ => Ok(Bytes::from(tail_bytes)),
        }
    }

    fn encode(&mut self, samples: &[f32]) -> Result<Bytes> {
        let bytes = match self.format.encoding {
            AudioEncoding::Pcm16 | AudioEncoding::WavPcm16 => pcm::pcm_bytes(samples),
            AudioEncoding::MuLaw => pcm::ulaw_bytes(samples),
            AudioEncoding::ALaw => pcm::alaw_bytes(samples),
            AudioEncoding::Mp3 | AudioEncoding::Opus => {
                self.compressed.as_mut().expect("compressed format has codec state").push(samples)?
            }
        };
        if self.format.encoding == AudioEncoding::WavPcm16 {
            self.wav_pcm.extend_from_slice(&bytes);
            Ok(Bytes::new())
        } else {
            Ok(Bytes::from(bytes))
        }
    }

    fn ensure_active(&self) -> Result<()> {
        if self.finished {
            return Err(AudioError::Finished);
        }
        Ok(())
    }
}
