use std::thread;

use tinfer_rust::audio::AudioFormat;

#[path = "audio_compressed/support.rs"]
mod support;

use support::{decode_mp3, decode_opus, encode, parse_ogg, reference_wave};

fn audio_format(value: &str) -> AudioFormat {
    value.parse().unwrap()
}

#[test]
fn declared_compressed_formats_produce_streams() {
    let samples = reference_wave(24_000);
    let formats = [
        audio_format("mp3_22050_32"),
        audio_format("mp3_24000_48"),
        audio_format("mp3_44100_32"),
        audio_format("mp3_44100_64"),
        audio_format("mp3_44100_96"),
        audio_format("mp3_44100_128"),
        audio_format("mp3_44100_192"),
        audio_format("opus_48000_32"),
        audio_format("opus_48000_64"),
        audio_format("opus_48000_96"),
        audio_format("opus_48000_128"),
        audio_format("opus_48000_192"),
    ];
    for format in formats {
        let output = encode(format, &[&samples]);
        assert!(!output.is_empty(), "{format:?}");
        if format.encoding == tinfer_rust::audio::AudioEncoding::Mp3 {
            let (decoded, rate, channels) = decode_mp3(&output);
            assert_eq!(rate, format.sample_rate as usize, "{format:?}");
            assert_eq!(channels, 1, "{format:?}");
            let first_audible = decoded.iter().position(|sample| sample.abs() > 500).unwrap();
            let last_audible = decoded.iter().rposition(|sample| sample.abs() > 500).unwrap();
            let seconds = (last_audible - first_audible + 1) as f64 / rate as f64;
            assert!((seconds - 1.0).abs() <= 0.03, "{format:?}: {seconds}");
            let rms = (decoded.iter().map(|sample| f64::from(*sample).powi(2)).sum::<f64>() / decoded.len() as f64).sqrt();
            assert!(rms > 1_000.0, "{format:?}: {rms}");
        }
    }
}

#[test]
fn mp3_bytes_do_not_depend_on_push_boundaries() {
    let samples = reference_wave(24_000);
    let whole = encode(audio_format("mp3_24000_48"), &[&samples]);
    let split = encode(audio_format("mp3_24000_48"), &[&samples[..1], &[], &samples[1..8_003], &samples[8_003..]]);
    assert_eq!(whole, split);
}

#[test]
fn compressed_encoders_are_response_scoped_and_thread_safe() {
    let handles: Vec<_> = (0..8)
        .map(|index| {
            thread::spawn(move || {
                let samples = reference_wave(24_000);
                let format = if index % 2 == 0 { audio_format("mp3_44100_96") } else { audio_format("opus_48000_96") };
                encode(format, &[&samples[..4_001], &samples[4_001..]])
            })
        })
        .collect();
    for output in handles.into_iter().map(|handle| handle.join().unwrap()) {
        assert!(!output.is_empty());
    }
}

#[test]
fn ogg_opus_pages_and_decoded_audio_match_the_stream_contract() {
    let samples = reference_wave(24_000);
    for format in [
        audio_format("opus_48000_32"),
        audio_format("opus_48000_64"),
        audio_format("opus_48000_96"),
        audio_format("opus_48000_128"),
        audio_format("opus_48000_192"),
    ] {
        let output = encode(format, &[&samples[..1], &samples[1..8_000], &samples[8_000..]]);
        let pages = parse_ogg(&output);
        assert_eq!(pages.iter().filter(|page| page.flags & 0x02 != 0).count(), 1);
        assert_eq!(pages.iter().filter(|page| page.flags & 0x04 != 0).count(), 1);
        assert_eq!(pages[0].packet.len(), 19);
        assert_eq!(&pages[0].packet[..8], b"OpusHead");
        assert_eq!(pages[0].packet[9], 1);
        assert_eq!(u32::from_le_bytes(pages[0].packet[12..16].try_into().unwrap()), 24_000);
        assert_eq!(&pages[1].packet[..8], b"OpusTags");
        assert!(pages.windows(2).all(|pair| pair[0].granule <= pair[1].granule));
        assert!(pages.iter().all(|page| page.serial == pages[0].serial));
        assert!(pages.iter().enumerate().all(|(index, page)| page.sequence == index as u32));
        let pre_skip = u16::from_le_bytes(pages[0].packet[10..12].try_into().unwrap()) as u64;
        let audio_pages = &pages[2..];
        for (index, page) in audio_pages[..audio_pages.len() - 1].iter().enumerate() {
            assert_eq!(page.granule, (index as u64 + 1) * 960);
        }
        assert_eq!(pages.last().unwrap().granule, pre_skip + 48_000);
        let (decoded, final_granule) = decode_opus(&output);
        assert_eq!(final_granule, pre_skip + 48_000);
        assert_eq!(decoded.len(), 48_000);
        let rms = (decoded.iter().map(|sample| sample.powi(2)).sum::<f32>() / decoded.len() as f32).sqrt();
        assert!(rms > 0.1, "{format:?}: {rms}");
        let crossings = decoded.windows(2).filter(|pair| pair[0].is_sign_negative() != pair[1].is_sign_negative()).count();
        assert!((crossings as i32 - 880).abs() < 30, "{format:?}: {crossings}");
    }
}

#[test]
fn empty_compressed_responses_have_a_terminal_stream() {
    let mp3 = encode(audio_format("mp3_24000_48"), &[&[]]);
    assert!(!mp3.is_empty());
    let opus = encode(audio_format("opus_48000_96"), &[&[]]);
    let pages = parse_ogg(&opus);
    assert_eq!(pages.iter().filter(|page| page.flags & 0x02 != 0).count(), 1);
    assert_eq!(pages.iter().filter(|page| page.flags & 0x04 != 0).count(), 1);
    assert_eq!(decode_opus(&opus).0.len(), 0);
}

#[test]
fn bitrate_controls_have_monotonic_size_sanity() {
    let samples = reference_wave(24_000);
    let mp3_sizes: Vec<_> = [
        audio_format("mp3_44100_32"),
        audio_format("mp3_44100_64"),
        audio_format("mp3_44100_96"),
        audio_format("mp3_44100_128"),
        audio_format("mp3_44100_192"),
    ]
    .map(|format| encode(format, &[&samples]).len())
    .to_vec();
    let opus_sizes: Vec<_> = [
        audio_format("opus_48000_32"),
        audio_format("opus_48000_64"),
        audio_format("opus_48000_96"),
        audio_format("opus_48000_128"),
        audio_format("opus_48000_192"),
    ]
    .map(|format| encode(format, &[&samples]).len())
    .to_vec();
    assert!(mp3_sizes.windows(2).all(|pair| pair[0] < pair[1]));
    assert!(opus_sizes.windows(2).all(|pair| pair[0] < pair[1]));
}

#[test]
fn production_ogg_responses_have_distinct_nonzero_serials() {
    let samples = reference_wave(24_000);
    let first = encode(audio_format("opus_48000_96"), &[&samples]);
    let second = encode(audio_format("opus_48000_96"), &[&samples]);
    let first_serial = parse_ogg(&first)[0].serial;
    let second_serial = parse_ogg(&second)[0].serial;
    assert_ne!(first_serial, 0);
    assert_ne!(second_serial, 0);
    assert_ne!(first_serial, second_serial);
}

#[test]
fn codec_state_survives_repeated_construction_and_drop() {
    let samples = reference_wave(24_000);
    for _ in 0..200 {
        assert!(!encode(audio_format("mp3_24000_48"), &[&samples[..97]]).is_empty());
        assert!(!encode(audio_format("opus_48000_64"), &[&samples[..97]]).is_empty());
    }
}

#[test]
fn empty_and_irregular_pushes_form_one_stream() {
    let samples = reference_wave(24_000);
    for format in [audio_format("mp3_24000_48"), audio_format("opus_48000_96")] {
        let output = encode(format, &[&[], &samples[..1], &samples[1..8001], &[], &samples[8001..]]);
        if format == audio_format("opus_48000_96") {
            assert_eq!(&output[..8], b"OggS\0\x02\0\0");
            assert_eq!(output.windows(8).filter(|part| *part == b"OpusHead").count(), 1);
            assert_eq!(output.windows(8).filter(|part| *part == b"OpusTags").count(), 1);
        } else {
            assert_eq!(output.windows(3).filter(|part| *part == b"ID3").count(), 0);
        }
    }
}
