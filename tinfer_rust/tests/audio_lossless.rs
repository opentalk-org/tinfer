use std::str::FromStr;

use bytes::Bytes;
use tinfer_rust::audio::{AudioEncoder, AudioError, AudioFormat};

fn audio_format(value: &str) -> AudioFormat {
    value.parse().unwrap()
}

fn encode(format: AudioFormat, source_rate: u32, chunks: &[&[f32]]) -> Vec<u8> {
    let mut encoder = AudioEncoder::new(format, source_rate).unwrap();
    let mut output = Vec::new();
    for chunk in chunks {
        output.extend_from_slice(&encoder.push(chunk).unwrap());
    }
    output.extend_from_slice(&encoder.finish().unwrap());
    output
}

fn fixture() -> serde_json::Value {
    serde_json::from_str(include_str!("fixtures/audio_golden.json")).unwrap()
}

fn reference_samples() -> Vec<f32> {
    serde_json::from_value(fixture()["samples"].clone()).unwrap()
}

#[test]
fn lossless_encodings_match_python_reference_bytes() {
    let fixture = fixture();
    let samples = reference_samples();
    for (format, key) in [(audio_format("pcm_16000"), "pcm"), (audio_format("ulaw_8000"), "ulaw"), (audio_format("alaw_8000"), "alaw")] {
        let expected: Vec<u8> = serde_json::from_value(fixture[key].clone()).unwrap();
        assert_eq!(encode(format, format.sample_rate, &[&samples]), expected);
    }
}

#[test]
fn wav_has_one_canonical_header_and_the_reference_pcm_payload() {
    let samples = reference_samples();
    let wav = encode(audio_format("wav_24000"), 24_000, &[&samples[..4], &samples[4..]]);
    let pcm: Vec<u8> = serde_json::from_value(fixture()["pcm"].clone()).unwrap();
    assert_eq!(&wav[0..4], b"RIFF");
    assert_eq!(u32::from_le_bytes(wav[4..8].try_into().unwrap()), 36 + pcm.len() as u32);
    assert_eq!(&wav[8..16], b"WAVEfmt ");
    assert_eq!(u16::from_le_bytes(wav[20..22].try_into().unwrap()), 1);
    assert_eq!(u16::from_le_bytes(wav[22..24].try_into().unwrap()), 1);
    assert_eq!(u32::from_le_bytes(wav[24..28].try_into().unwrap()), 24_000);
    assert_eq!(u32::from_le_bytes(wav[28..32].try_into().unwrap()), 48_000);
    assert_eq!(u16::from_le_bytes(wav[32..34].try_into().unwrap()), 2);
    assert_eq!(u16::from_le_bytes(wav[34..36].try_into().unwrap()), 16);
    assert_eq!(&wav[36..40], b"data");
    assert_eq!(u32::from_le_bytes(wav[40..44].try_into().unwrap()), pcm.len() as u32);
    assert_eq!(&wav[44..], pcm);
    assert_eq!(wav.windows(4).filter(|value| *value == b"RIFF").count(), 1);
}

#[test]
fn parsing_is_exact_and_covers_the_declared_speech_formats() {
    let formats = [
        "mp3_22050_32",
        "mp3_24000_48",
        "mp3_44100_32",
        "mp3_44100_64",
        "mp3_44100_96",
        "mp3_44100_128",
        "mp3_44100_192",
        "pcm_8000",
        "pcm_16000",
        "pcm_22050",
        "pcm_24000",
        "pcm_32000",
        "pcm_44100",
        "pcm_48000",
        "wav_8000",
        "wav_16000",
        "wav_22050",
        "wav_24000",
        "wav_32000",
        "wav_44100",
        "wav_48000",
        "ulaw_8000",
        "alaw_8000",
        "opus_48000_32",
        "opus_48000_64",
        "opus_48000_96",
        "opus_48000_128",
        "opus_48000_192",
    ];
    for value in formats {
        let parsed = AudioFormat::from_str(value).unwrap();
        assert_ne!(parsed.sample_rate, 0);
    }
    for value in ["PCM_24000", " pcm_24000", "pcm_24000 ", "pcm-whatever-24000", ""] {
        assert!(matches!(AudioFormat::from_str(value), Err(AudioError::UnknownFormat(_))));
    }
    assert_eq!(audio_format("mp3_24000_48").bitrate_kbps, Some(48));
}

#[test]
fn construction_rejects_bad_source_rates() {
    assert!(matches!(AudioEncoder::new(audio_format("pcm_24000"), 0), Err(AudioError::UnsupportedSampleRate(0))));
    assert!(matches!(AudioEncoder::new(audio_format("pcm_24000"), 12_345), Err(AudioError::UnsupportedSampleRate(12_345))));
}

#[test]
fn lifecycle_and_sample_failures_are_explicit_and_atomic() {
    let mut encoder = AudioEncoder::new(audio_format("pcm_24000"), 24_000).unwrap();
    assert_eq!(encoder.push(&[]).unwrap(), Bytes::new());
    for sample in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        assert_eq!(encoder.push(&[0.25, sample]), Err(AudioError::NonFiniteSample));
    }
    assert_eq!(encoder.push(&[0.5]).unwrap().len(), 2);
    assert!(encoder.finish().unwrap().is_empty());
    assert_eq!(encoder.push(&[0.0]), Err(AudioError::Finished));
    assert_eq!(encoder.finish(), Err(AudioError::Finished));
}

#[test]
fn arbitrary_chunking_is_byte_identical_for_every_lossless_format() {
    let samples: Vec<f32> = (0..5_123).map(|index| ((index as f32 * 0.071).sin() * 0.7).clamp(-1.0, 1.0)).collect();
    let one_sample: Vec<&[f32]> = samples.chunks(1).collect();
    let irregular: Vec<&[f32]> = samples.chunks(137).collect();
    for format in [audio_format("pcm_8000"), audio_format("ulaw_8000"), audio_format("alaw_8000"), audio_format("wav_44100")] {
        let whole = encode(format, 24_000, &[&samples]);
        assert_eq!(encode(format, 24_000, &one_sample), whole);
        assert_eq!(encode(format, 24_000, &irregular), whole);
    }
}

#[test]
fn resampling_clamps_finite_input_before_fft_processing() {
    let samples: Vec<f32> = (0..2_057)
        .map(|index| match index % 6 {
            0 => f32::MAX,
            1 => f32::MIN,
            2 => 4.0,
            3 => -2.0,
            4 => 0.25,
            _ => -0.5,
        })
        .collect();
    let clamped: Vec<f32> = samples.iter().map(|sample| sample.clamp(-1.0, 1.0)).collect();
    let expected = encode(audio_format("pcm_8000"), 24_000, &[&clamped]);
    let one_sample: Vec<&[f32]> = samples.chunks(1).collect();
    let irregular: Vec<&[f32]> = samples.chunks(137).collect();

    assert_eq!(encode(audio_format("pcm_8000"), 24_000, &[&samples]), expected);
    assert_eq!(encode(audio_format("pcm_8000"), 24_000, &one_sample), expected);
    assert_eq!(encode(audio_format("pcm_8000"), 24_000, &irregular), expected);
    assert!(expected.chunks_exact(2).all(|bytes| {
        let sample = i16::from_le_bytes(bytes.try_into().unwrap());
        (-32_767..=32_767).contains(&sample)
    }));
}

#[test]
fn rejected_resampled_push_does_not_mutate_pending_input() {
    let prefix = vec![0.25; 600];
    let suffix = vec![-0.5; 1_500];
    let invalid = [f32::MAX, 2.0, f32::NAN];
    let expected = encode(audio_format("pcm_8000"), 24_000, &[&prefix, &suffix]);
    let mut encoder = AudioEncoder::new(audio_format("pcm_8000"), 24_000).unwrap();
    let mut actual = encoder.push(&prefix).unwrap().to_vec();

    assert_eq!(encoder.push(&invalid), Err(AudioError::NonFiniteSample));
    actual.extend_from_slice(&encoder.push(&suffix).unwrap());
    actual.extend_from_slice(&encoder.finish().unwrap());
    assert_eq!(actual, expected);
}

#[test]
fn wav_buffers_pushes_and_emits_one_file_only_at_finish() {
    let mut encoder = AudioEncoder::new(audio_format("wav_16000"), 24_000).unwrap();
    assert!(encoder.push(&[0.1; 2_049]).unwrap().is_empty());
    assert!(encoder.push(&[]).unwrap().is_empty());
    let wav = encoder.finish().unwrap();
    assert_eq!(&wav[..4], b"RIFF");
    assert_eq!(wav.windows(4).filter(|value| *value == b"RIFF").count(), 1);
}

#[test]
fn resampling_has_reference_length_without_long_stream_drift() {
    let samples: Vec<f32> = (0..240_017).map(|index| (index as f32 * 440.0 * std::f32::consts::TAU / 24_000.0).sin() * 0.2).collect();
    for target in [8_000_u32, 16_000, 22_050, 24_000, 32_000, 44_100, 48_000] {
        let format = match target {
            8_000 => audio_format("pcm_8000"),
            16_000 => audio_format("pcm_16000"),
            22_050 => audio_format("pcm_22050"),
            24_000 => audio_format("pcm_24000"),
            32_000 => audio_format("pcm_32000"),
            44_100 => audio_format("pcm_44100"),
            48_000 => audio_format("pcm_48000"),
            _ => unreachable!(),
        };
        let bytes = encode(format, 24_000, &[&samples]);
        let expected = (samples.len() as u64 * u64::from(target)).div_ceil(24_000) as usize;
        assert_eq!(bytes.len(), expected * 2, "target rate {target}");
        let quantized: Vec<f32> =
            bytes.chunks_exact(2).map(|sample| i16::from_le_bytes(sample.try_into().unwrap()) as f32 / 32_767.0).collect();
        let rms = (quantized.iter().map(|value| value * value).sum::<f32>() / quantized.len() as f32).sqrt();
        // librosa's soxr reference is 0.14142; FFT windowing differs at the stream edges.
        assert!((rms - 0.14142).abs() < 0.002, "target rate {target}: {rms}");
    }
}

#[test]
fn short_and_partial_resampling_lengths_match_librosa_ceil_contract() {
    for length in [0_usize, 1, 2, 3, 10, 1_023, 1_024, 1_025] {
        let samples = vec![0.1; length];
        for (format, target) in [
            (audio_format("pcm_8000"), 8_000_u64),
            (audio_format("pcm_16000"), 16_000),
            (audio_format("pcm_22050"), 22_050),
            (audio_format("pcm_32000"), 32_000),
            (audio_format("pcm_44100"), 44_100),
            (audio_format("pcm_48000"), 48_000),
        ] {
            let bytes = encode(format, 24_000, &[&samples]);
            let expected = (length as u64 * target).div_ceil(24_000) as usize;
            assert_eq!(bytes.len(), expected * 2, "input {length}, target {target}");
        }
    }
}
