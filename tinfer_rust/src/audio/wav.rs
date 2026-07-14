use super::{AudioError, Result};

const RIFF_OVERHEAD: u64 = 36;

pub(crate) fn encode(sample_rate: u32, pcm: &[u8]) -> Result<Vec<u8>> {
    let payload_size = u32::try_from(pcm.len()).map_err(|_| AudioError::WavTooLarge)?;
    let riff_size = u32::try_from(RIFF_OVERHEAD + pcm.len() as u64).map_err(|_| AudioError::WavTooLarge)?;
    let byte_rate = sample_rate.checked_mul(2).ok_or(AudioError::WavTooLarge)?;
    let mut wav = Vec::with_capacity(44 + pcm.len());
    wav.extend_from_slice(b"RIFF");
    wav.extend_from_slice(&riff_size.to_le_bytes());
    wav.extend_from_slice(b"WAVEfmt ");
    wav.extend_from_slice(&16_u32.to_le_bytes());
    wav.extend_from_slice(&1_u16.to_le_bytes());
    wav.extend_from_slice(&1_u16.to_le_bytes());
    wav.extend_from_slice(&sample_rate.to_le_bytes());
    wav.extend_from_slice(&byte_rate.to_le_bytes());
    wav.extend_from_slice(&2_u16.to_le_bytes());
    wav.extend_from_slice(&16_u16.to_le_bytes());
    wav.extend_from_slice(b"data");
    wav.extend_from_slice(&payload_size.to_le_bytes());
    wav.extend_from_slice(pcm);
    Ok(wav)
}
