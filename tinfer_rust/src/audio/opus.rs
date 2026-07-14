use std::collections::VecDeque;
use std::ffi::{c_float, c_int, c_uchar, c_void};
use std::sync::OnceLock;
use std::sync::atomic::{AtomicU32, Ordering};

use libloading::Library;

use super::compressed::check_native;
use super::ogg::OggStream;
use super::{AudioError, AudioFormat, Codec, Result};

const FRAME_SAMPLES: usize = 960;
const MAX_PACKET_BYTES: usize = 1_275;
const OPUS_APPLICATION_AUDIO: c_int = 2049;
const OPUS_SET_BITRATE_REQUEST: c_int = 4002;
const OPUS_GET_LOOKAHEAD_REQUEST: c_int = 4027;
const OGG_BOS: u8 = 0x02;
const OGG_EOS: u8 = 0x04;

type Create = unsafe extern "C" fn(c_int, c_int, c_int, *mut c_int) -> *mut c_void;
type Destroy = unsafe extern "C" fn(*mut c_void);
type Encode = unsafe extern "C" fn(*mut c_void, *const c_float, c_int, *mut c_uchar, c_int) -> c_int;
type Ctl = unsafe extern "C" fn(*mut c_void, c_int, ...) -> c_int;

struct OpusApi {
    _library: Library,
    create: Create,
    destroy: Destroy,
    encode: Encode,
    ctl: Ctl,
}

// Function pointers stay valid because the owning library is retained for the process lifetime.
unsafe impl Send for OpusApi {}
unsafe impl Sync for OpusApi {}

pub(crate) struct OpusEncoder {
    api: &'static OpusApi,
    state: *mut c_void,
    stream: OggStream,
    header: Option<Vec<u8>>,
    pending_pcm: VecDeque<f32>,
    pending_packet: Option<Vec<u8>>,
    pre_skip: u16,
    encoded_samples: u64,
    logical_samples: u64,
    finished: bool,
}

unsafe impl Send for OpusEncoder {}

impl OpusEncoder {
    pub(crate) fn new(format: AudioFormat, source_rate: u32) -> Result<Self> {
        Self::with_serial(format, source_rate, next_serial())
    }

    fn with_serial(format: AudioFormat, source_rate: u32, serial: u32) -> Result<Self> {
        let api = opus_api()?;
        let mut code = 0;
        let state = unsafe { (api.create)(48_000, 1, OPUS_APPLICATION_AUDIO, &mut code) };
        check_native(Codec::Opus, code)?;
        if state.is_null() {
            return Err(AudioError::CodecInitialization(Codec::Opus));
        }
        let mut encoder = Self {
            api,
            state,
            stream: OggStream::new(serial),
            header: None,
            pending_pcm: VecDeque::new(),
            pending_packet: None,
            pre_skip: 0,
            encoded_samples: 0,
            logical_samples: 0,
            finished: false,
        };
        encoder.configure(format)?;
        encoder.header = Some(encoder.headers(source_rate)?);
        Ok(encoder)
    }

    fn configure(&mut self, format: AudioFormat) -> Result<()> {
        let bitrate = c_int::from(format.bitrate_kbps.expect("Opus format has a bitrate")) * 1_000;
        check_native(Codec::Opus, unsafe { (self.api.ctl)(self.state, OPUS_SET_BITRATE_REQUEST, bitrate) })?;
        let mut lookahead: c_int = 0;
        check_native(Codec::Opus, unsafe { (self.api.ctl)(self.state, OPUS_GET_LOOKAHEAD_REQUEST, &mut lookahead as *mut c_int) })?;
        self.pre_skip = lookahead as u16;
        Ok(())
    }

    pub(crate) fn headers(&mut self, original_rate: u32) -> Result<Vec<u8>> {
        let mut head = Vec::with_capacity(19);
        head.extend_from_slice(b"OpusHead");
        head.extend_from_slice(&[1, 1]);
        head.extend_from_slice(&self.pre_skip.to_le_bytes());
        head.extend_from_slice(&original_rate.to_le_bytes());
        head.extend_from_slice(&0_i16.to_le_bytes());
        head.push(0);
        let mut output = self.stream.packet(&head, 0, OGG_BOS)?;
        let vendor = b"tinfer-rust";
        let mut tags = Vec::with_capacity(12 + vendor.len());
        tags.extend_from_slice(b"OpusTags");
        tags.extend_from_slice(&(vendor.len() as u32).to_le_bytes());
        tags.extend_from_slice(vendor);
        tags.extend_from_slice(&0_u32.to_le_bytes());
        output.extend(self.stream.packet(&tags, 0, 0)?);
        Ok(output)
    }

    pub(crate) fn push(&mut self, samples: &[f32]) -> Result<Vec<u8>> {
        self.logical_samples += samples.len() as u64;
        self.pending_pcm.extend(samples.iter().copied());
        let mut output = self.header.take().unwrap_or_default();
        while self.pending_pcm.len() >= FRAME_SAMPLES {
            let frame: Vec<f32> = self.pending_pcm.drain(..FRAME_SAMPLES).collect();
            self.queue_frame(&frame, &mut output)?;
        }
        Ok(output)
    }

    pub(crate) fn finish(&mut self) -> Result<Vec<u8>> {
        assert!(!self.finished, "AudioEncoder guards compressed encoder lifecycle");
        let mut output = self.header.take().unwrap_or_default();
        if !self.pending_pcm.is_empty() {
            let mut frame: Vec<f32> = self.pending_pcm.drain(..).collect();
            frame.resize(FRAME_SAMPLES, 0.0);
            self.queue_frame(&frame, &mut output)?;
        }
        let required_samples = u64::from(self.pre_skip) + self.logical_samples;
        while self.encoded_samples < required_samples || self.pending_packet.is_none() {
            self.queue_frame(&[0.0; FRAME_SAMPLES], &mut output)?;
        }
        let packet = self.pending_packet.take().expect("finish always has a terminal Opus packet");
        let final_granule = required_samples;
        output.extend(self.stream.packet(&packet, final_granule, OGG_EOS)?);
        self.finished = true;
        Ok(output)
    }

    fn queue_frame(&mut self, frame: &[f32], output: &mut Vec<u8>) -> Result<()> {
        let mut packet = vec![0_u8; MAX_PACKET_BYTES];
        let written = unsafe {
            (self.api.encode)(self.state, frame.as_ptr(), FRAME_SAMPLES as c_int, packet.as_mut_ptr(), MAX_PACKET_BYTES as c_int)
        };
        let written = check_native(Codec::Opus, written)? as usize;
        packet.truncate(written);
        self.encoded_samples += FRAME_SAMPLES as u64;
        if let Some(previous) = self.pending_packet.replace(packet) {
            let granule = self.encoded_samples - FRAME_SAMPLES as u64;
            output.extend(self.stream.packet(&previous, granule, 0)?);
        }
        Ok(())
    }
}

impl Drop for OpusEncoder {
    fn drop(&mut self) {
        unsafe { (self.api.destroy)(self.state) };
    }
}

fn next_serial() -> u32 {
    static NEXT: AtomicU32 = AtomicU32::new(1);
    NEXT.fetch_add(1, Ordering::Relaxed)
}

fn opus_api() -> Result<&'static OpusApi> {
    static API: OnceLock<std::result::Result<OpusApi, String>> = OnceLock::new();
    API.get_or_init(load_opus).as_ref().map_err(|reason| AudioError::CodecUnavailable { codec: Codec::Opus, reason: reason.clone() })
}

fn load_opus() -> std::result::Result<OpusApi, String> {
    unsafe {
        let library = Library::new("libopus.so.0").map_err(|error| error.to_string())?;
        macro_rules! symbol {
            ($name:literal, $type:ty) => {
                *library.get::<$type>($name).map_err(|error| error.to_string())?
            };
        }
        Ok(OpusApi {
            create: symbol!(b"opus_encoder_create\0", Create),
            destroy: symbol!(b"opus_encoder_destroy\0", Destroy),
            encode: symbol!(b"opus_encode_float\0", Encode),
            ctl: symbol!(b"opus_encoder_ctl\0", Ctl),
            _library: library,
        })
    }
}
