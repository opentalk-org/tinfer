use std::ffi::{c_float, c_int, c_uchar, c_void};
use std::sync::OnceLock;

use libloading::Library;

use super::compressed::check_native;
use super::{AudioError, AudioFormat, Codec, Result};

type Lame = *mut c_void;
type Init = unsafe extern "C" fn() -> Lame;
type Close = unsafe extern "C" fn(Lame) -> c_int;
type SetInt = unsafe extern "C" fn(Lame, c_int) -> c_int;
type InitParams = unsafe extern "C" fn(Lame) -> c_int;
type Encode = unsafe extern "C" fn(Lame, *const c_float, *const c_float, c_int, *mut c_uchar, c_int) -> c_int;
type Flush = unsafe extern "C" fn(Lame, *mut c_uchar, c_int) -> c_int;

struct LameApi {
    _library: Library,
    init: Init,
    close: Close,
    set_in_rate: SetInt,
    set_out_rate: SetInt,
    set_channels: SetInt,
    set_mode: SetInt,
    set_bitrate: SetInt,
    init_params: InitParams,
    encode: Encode,
    flush: Flush,
}

// Function pointers stay valid because the owning library is retained for the process lifetime.
unsafe impl Send for LameApi {}
unsafe impl Sync for LameApi {}

pub(crate) struct Mp3Encoder {
    api: &'static LameApi,
    state: Lame,
    finished: bool,
}

unsafe impl Send for Mp3Encoder {}

impl Mp3Encoder {
    pub(crate) fn new(format: AudioFormat) -> Result<Self> {
        let api = lame_api()?;
        let state = unsafe { (api.init)() };
        if state.is_null() {
            return Err(AudioError::CodecInitialization(Codec::Mp3));
        }
        let mut encoder = Self { api, state, finished: false };
        encoder.configure(format)?;
        Ok(encoder)
    }

    fn configure(&mut self, format: AudioFormat) -> Result<()> {
        self.set(self.api.set_in_rate, format.sample_rate as c_int)?;
        self.set(self.api.set_out_rate, format.sample_rate as c_int)?;
        self.set(self.api.set_channels, 1)?;
        self.set(self.api.set_mode, 3)?;
        self.set(self.api.set_bitrate, c_int::from(format.bitrate_kbps.expect("MP3 format has a bitrate")))?;
        check_native(Codec::Mp3, unsafe { (self.api.init_params)(self.state) }).map(drop)
    }

    pub(crate) fn push(&mut self, samples: &[f32]) -> Result<Vec<u8>> {
        let count = c_int::try_from(samples.len()).map_err(|_| AudioError::AudioTooLong)?;
        let capacity = samples.len() * 5 / 4 + 7_200;
        let capacity_int = c_int::try_from(capacity).map_err(|_| AudioError::AudioTooLong)?;
        let mut output = vec![0_u8; capacity];
        let written =
            unsafe { (self.api.encode)(self.state, samples.as_ptr(), std::ptr::null(), count, output.as_mut_ptr(), capacity_int) };
        let written = check_native(Codec::Mp3, written)? as usize;
        output.truncate(written);
        Ok(output)
    }

    pub(crate) fn finish(&mut self) -> Result<Vec<u8>> {
        assert!(!self.finished, "AudioEncoder guards compressed encoder lifecycle");
        let mut output = vec![0_u8; 7_200];
        let written = unsafe { (self.api.flush)(self.state, output.as_mut_ptr(), 7_200) };
        let written = check_native(Codec::Mp3, written)? as usize;
        output.truncate(written);
        self.finished = true;
        Ok(output)
    }

    fn set(&self, function: SetInt, value: c_int) -> Result<()> {
        check_native(Codec::Mp3, unsafe { function(self.state, value) }).map(drop)
    }
}

impl Drop for Mp3Encoder {
    fn drop(&mut self) {
        let _ = unsafe { (self.api.close)(self.state) };
    }
}

fn lame_api() -> Result<&'static LameApi> {
    static API: OnceLock<std::result::Result<LameApi, String>> = OnceLock::new();
    API.get_or_init(load_lame).as_ref().map_err(|reason| AudioError::CodecUnavailable { codec: Codec::Mp3, reason: reason.clone() })
}

fn load_lame() -> std::result::Result<LameApi, String> {
    unsafe {
        let library = Library::new("libmp3lame.so.0").map_err(|error| error.to_string())?;
        macro_rules! symbol {
            ($name:literal, $type:ty) => {
                *library.get::<$type>($name).map_err(|error| error.to_string())?
            };
        }
        Ok(LameApi {
            init: symbol!(b"lame_init\0", Init),
            close: symbol!(b"lame_close\0", Close),
            set_in_rate: symbol!(b"lame_set_in_samplerate\0", SetInt),
            set_out_rate: symbol!(b"lame_set_out_samplerate\0", SetInt),
            set_channels: symbol!(b"lame_set_num_channels\0", SetInt),
            set_mode: symbol!(b"lame_set_mode\0", SetInt),
            set_bitrate: symbol!(b"lame_set_brate\0", SetInt),
            init_params: symbol!(b"lame_init_params\0", InitParams),
            encode: symbol!(b"lame_encode_buffer_ieee_float\0", Encode),
            flush: symbol!(b"lame_encode_flush\0", Flush),
            _library: library,
        })
    }
}
