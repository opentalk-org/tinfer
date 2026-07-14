pub(crate) fn quantize(samples: &[f32]) -> Vec<i16> {
    samples.iter().map(|sample| (sample.clamp(-1.0, 1.0) * 32_767.0) as i16).collect()
}

pub(crate) fn pcm_bytes(samples: &[f32]) -> Vec<u8> {
    quantize(samples).into_iter().flat_map(i16::to_le_bytes).collect()
}

pub(crate) fn ulaw_bytes(samples: &[f32]) -> Vec<u8> {
    quantize(samples).into_iter().map(linear_to_ulaw).collect()
}

pub(crate) fn alaw_bytes(samples: &[f32]) -> Vec<u8> {
    quantize(samples).into_iter().map(linear_to_alaw).collect()
}

fn linear_to_ulaw(sample: i16) -> u8 {
    const BIAS: i32 = 0x84 >> 2;
    const CLIP: i32 = 32_635 >> 2;
    let mut value = i32::from(sample) >> 2;
    let mask = if value < 0 { 0x7f } else { 0xff };
    if value < 0 {
        value = -value;
    }
    value = value.min(CLIP) + BIAS;
    let exponent = (31 - value.leading_zeros() - 5).min(7) as u8;
    let mantissa = ((value >> (u32::from(exponent) + 1)) & 0x0f) as u8;
    ((exponent << 4) | mantissa) ^ mask
}

fn linear_to_alaw(sample: i16) -> u8 {
    let mut value = i32::from(sample) >> 3;
    let mask = if value >= 0 { 0xd5 } else { 0x55 };
    if value < 0 {
        value = -value - 1;
    }
    let segment = match value {
        0..=0x1f => 0,
        0x20..=0x3f => 1,
        0x40..=0x7f => 2,
        0x80..=0xff => 3,
        0x100..=0x1ff => 4,
        0x200..=0x3ff => 5,
        0x400..=0x7ff => 6,
        _ => 7,
    };
    let mantissa = if segment < 2 { (value >> 1) & 0x0f } else { (value >> segment) & 0x0f };
    ((segment << 4) | mantissa) as u8 ^ mask
}
