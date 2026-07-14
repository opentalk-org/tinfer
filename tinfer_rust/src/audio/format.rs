use std::str::FromStr;

use super::{AudioError, Result};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AudioEncoding {
    Pcm16,
    WavPcm16,
    MuLaw,
    ALaw,
    Mp3,
    Opus,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct AudioFormat {
    pub encoding: AudioEncoding,
    pub sample_rate: u32,
    pub bitrate_kbps: Option<u16>,
}

impl AudioFormat {
    const fn new(encoding: AudioEncoding, sample_rate: u32, bitrate_kbps: Option<u16>) -> Self {
        Self { encoding, sample_rate, bitrate_kbps }
    }
}

impl FromStr for AudioFormat {
    type Err = AudioError;

    fn from_str(value: &str) -> Result<Self> {
        let parts = value.split('_').collect::<Vec<_>>();
        let parsed = match parts.as_slice() {
            [encoding, rate] => (*encoding, rate.parse().ok(), None),
            [encoding, rate, bitrate] => (*encoding, rate.parse().ok(), bitrate.parse().ok()),
            _ => return Err(AudioError::UnknownFormat(value.into())),
        };
        let format = match parsed {
            ("pcm", Some(rate @ (8_000 | 16_000 | 22_050 | 24_000 | 32_000 | 44_100 | 48_000)), None) => {
                Self::new(AudioEncoding::Pcm16, rate, None)
            }
            ("wav", Some(rate @ (8_000 | 16_000 | 22_050 | 24_000 | 32_000 | 44_100 | 48_000)), None) => {
                Self::new(AudioEncoding::WavPcm16, rate, None)
            }
            ("ulaw", Some(8_000), None) => Self::new(AudioEncoding::MuLaw, 8_000, None),
            ("alaw", Some(8_000), None) => Self::new(AudioEncoding::ALaw, 8_000, None),
            ("mp3", Some(22_050), Some(32)) => Self::new(AudioEncoding::Mp3, 22_050, Some(32)),
            ("mp3", Some(24_000), Some(48)) => Self::new(AudioEncoding::Mp3, 24_000, Some(48)),
            ("mp3", Some(44_100), Some(rate @ (32 | 64 | 96 | 128 | 192))) => Self::new(AudioEncoding::Mp3, 44_100, Some(rate)),
            ("opus", Some(48_000), Some(rate @ (32 | 64 | 96 | 128 | 192))) => Self::new(AudioEncoding::Opus, 48_000, Some(rate)),
            _ => return Err(AudioError::UnknownFormat(value.into())),
        };
        Ok(format)
    }
}
