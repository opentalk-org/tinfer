use std::f32::consts::TAU;
use std::ffi::{c_float, c_int, c_uchar, c_void};
use std::io::Cursor;

use libloading::Library;
use minimp3::{Decoder, Error as Mp3Error};
use tinfer_rust::audio::{AudioEncoder, AudioFormat};

pub fn reference_wave(sample_rate: u32) -> Vec<f32> {
    (0..sample_rate).map(|index| 0.35 * (TAU * 440.0 * index as f32 / sample_rate as f32).sin()).collect()
}

pub fn decode_mp3(bytes: &[u8]) -> (Vec<i16>, usize, usize) {
    let mut decoder = Decoder::new(Cursor::new(bytes));
    let mut samples = Vec::new();
    let mut sample_rate: usize = 0;
    let mut channels: usize = 0;
    loop {
        match decoder.next_frame() {
            Ok(frame) => {
                sample_rate = frame.sample_rate as usize;
                channels = frame.channels;
                samples.extend(frame.data);
            }
            Err(Mp3Error::Eof) => break,
            Err(error) => panic!("MP3 decode failed: {error:?}"),
        }
    }
    (samples, sample_rate, channels)
}

pub struct OggPage<'a> {
    pub flags: u8,
    pub granule: u64,
    pub serial: u32,
    pub sequence: u32,
    pub packet: &'a [u8],
}

fn ogg_crc(bytes: &[u8]) -> u32 {
    let mut checksum = 0_u32;
    for byte in bytes {
        checksum ^= u32::from(*byte) << 24;
        for _ in 0..8 {
            checksum = if checksum & 0x8000_0000 != 0 { (checksum << 1) ^ 0x04c1_1db7 } else { checksum << 1 };
        }
    }
    checksum
}

pub fn parse_ogg(bytes: &[u8]) -> Vec<OggPage<'_>> {
    let mut pages = Vec::new();
    let mut offset = 0;
    while offset < bytes.len() {
        assert_eq!(&bytes[offset..offset + 4], b"OggS");
        let segments = bytes[offset + 26] as usize;
        let body_len: usize = bytes[offset + 27..offset + 27 + segments].iter().map(|value| *value as usize).sum();
        let end = offset + 27 + segments + body_len;
        let expected = u32::from_le_bytes(bytes[offset + 22..offset + 26].try_into().unwrap());
        let mut page_bytes = bytes[offset..end].to_vec();
        page_bytes[22..26].fill(0);
        assert_eq!(ogg_crc(&page_bytes), expected);
        pages.push(OggPage {
            flags: bytes[offset + 5],
            granule: u64::from_le_bytes(bytes[offset + 6..offset + 14].try_into().unwrap()),
            serial: u32::from_le_bytes(bytes[offset + 14..offset + 18].try_into().unwrap()),
            sequence: u32::from_le_bytes(bytes[offset + 18..offset + 22].try_into().unwrap()),
            packet: &bytes[offset + 27 + segments..end],
        });
        offset = end;
    }
    pages
}

pub fn decode_opus(bytes: &[u8]) -> (Vec<f32>, u64) {
    type Create = unsafe extern "C" fn(c_int, c_int, *mut c_int) -> *mut c_void;
    type Decode = unsafe extern "C" fn(*mut c_void, *const c_uchar, c_int, *mut c_float, c_int, c_int) -> c_int;
    type Destroy = unsafe extern "C" fn(*mut c_void);
    let pages = parse_ogg(bytes);
    let pre_skip = u16::from_le_bytes(pages[0].packet[10..12].try_into().unwrap()) as usize;
    let final_granule = pages.last().unwrap().granule;
    let library = unsafe { Library::new("libopus.so.0") }.unwrap();
    let create = unsafe { *library.get::<Create>(b"opus_decoder_create\0").unwrap() };
    let decode = unsafe { *library.get::<Decode>(b"opus_decode_float\0").unwrap() };
    let destroy = unsafe { *library.get::<Destroy>(b"opus_decoder_destroy\0").unwrap() };
    let mut error = 0;
    let decoder = unsafe { create(48_000, 1, &mut error) };
    assert_eq!(error, 0);
    assert!(!decoder.is_null());
    let mut decoded = Vec::new();
    for page in &pages[2..] {
        let mut frame = vec![0.0; 5_760];
        let count =
            unsafe { decode(decoder, page.packet.as_ptr(), page.packet.len() as c_int, frame.as_mut_ptr(), frame.len() as c_int, 0) };
        assert!(count > 0, "Opus decoder code {count}");
        frame.truncate(count as usize);
        decoded.extend(frame);
    }
    unsafe { destroy(decoder) };
    let logical = (final_granule as usize).checked_sub(pre_skip).unwrap();
    decoded.drain(..pre_skip);
    decoded.truncate(logical);
    (decoded, final_granule)
}

pub fn encode(format: AudioFormat, chunks: &[&[f32]]) -> Vec<u8> {
    let mut encoder = AudioEncoder::new(format, 24_000).unwrap();
    let mut output = Vec::new();
    for chunk in chunks {
        output.extend_from_slice(&encoder.push(chunk).unwrap());
    }
    output.extend_from_slice(&encoder.finish().unwrap());
    output
}
