use crate::{AlignrustError, EngineConfig};
use libc::{c_char, c_int, c_void};
use std::collections::BTreeMap;
use std::ffi::{CStr, CString};

#[repr(C)]
#[allow(non_camel_case_types)]
enum espeak_AUDIO_OUTPUT {
    AUDIO_OUTPUT_PLAYBACK = 0,
    AUDIO_OUTPUT_RETRIEVAL = 1,
    AUDIO_OUTPUT_SYNCHRONOUS = 2,
    AUDIO_OUTPUT_SYNCH_PLAYBACK = 3,
}

#[repr(C)]
#[allow(non_camel_case_types)]
#[derive(Copy, Clone, PartialEq, Eq)]
enum espeak_ERROR {
    EE_OK = 0,
    EE_INTERNAL_ERROR = -1,
    EE_BUFFER_FULL = 1,
    EE_NOT_FOUND = 2,
}

#[repr(C)]
pub struct espeak_VOICE {
    pub name: *const c_char,
    pub languages: *const c_char,
    pub identifier: *const c_char,
    pub gender: u8,
    pub age: u8,
    pub variant: u8,
    pub xx1: u8,
    pub score: c_int,
    pub spare: *mut c_void,
}

unsafe extern "C" {
    fn espeak_Initialize(
        output: espeak_AUDIO_OUTPUT,
        buflength: c_int,
        path: *const c_char,
        options: c_int,
    ) -> c_int;
    fn espeak_ListVoices(voice_spec: *mut espeak_VOICE) -> *const *const espeak_VOICE;
    fn espeak_SetVoiceByName(name: *const c_char) -> espeak_ERROR;
    fn espeak_TextToPhonemes(
        textptr: *mut *const c_void,
        textmode: c_int,
        phonememode: c_int,
    ) -> *const c_char;
}

pub struct EspeakPhonemizer {
    tie: bool,
}

impl EspeakPhonemizer {
    pub fn new(cfg: &EngineConfig) -> Result<Self, AlignrustError> {
        let sr = unsafe { espeak_Initialize(espeak_AUDIO_OUTPUT::AUDIO_OUTPUT_SYNCHRONOUS, 0, std::ptr::null(), 0) };
        if sr <= 0 {
            return Err(AlignrustError::Message("espeak_Initialize failed".to_owned()));
        }

        let voices = unsafe { espeak_ListVoices(std::ptr::null_mut()) };
        if voices.is_null() {
            return Err(AlignrustError::Message(
                "espeak_ListVoices failed".to_owned(),
            ));
        }

        let mut available: BTreeMap<String, String> = BTreeMap::new();
        let mut i = 0isize;
        loop {
            let vp = unsafe { *voices.offset(i) };
            if vp.is_null() {
                break;
            }
            let v = unsafe { &*vp };
            let lang_ptr = v.languages;
            if lang_ptr.is_null() {
                i += 1;
                continue;
            }
            let lang_bytes = unsafe { CStr::from_ptr(lang_ptr) }.to_bytes();
            if lang_bytes.len() < 2 {
                i += 1;
                continue;
            }
            let mut lang = String::new();
            let mut j = 1usize;
            while j < lang_bytes.len() && lang_bytes[j] != 0 {
                lang.push(lang_bytes[j] as char);
                j += 1;
            }
            if !lang.is_empty() && !available.contains_key(&lang) {
                let ident = if v.identifier.is_null() {
                    ""
                } else {
                    unsafe { CStr::from_ptr(v.identifier) }.to_str().unwrap_or("")
                };
                available.insert(lang, ident.to_owned());
            }
            i += 1;
        }

        let mut voice_name = cfg.language.clone();
        if let Some(mapped) = available.get(&cfg.language) {
            if !mapped.is_empty() {
                voice_name = mapped.clone();
            }
        }

        let voice_c = CString::new(voice_name.clone()).map_err(|_| {
            AlignrustError::Message("voice name contained NUL byte".to_owned())
        })?;
        let ok = unsafe { espeak_SetVoiceByName(voice_c.as_ptr()) };
        if ok != espeak_ERROR::EE_OK {
            return Err(AlignrustError::Message(format!(
                "failed to load voice: {}",
                cfg.language
            )));
        }

        Ok(Self { tie: cfg.tie })
    }

    pub fn text_to_phonemes(&self, text: &str) -> Result<String, AlignrustError> {
        let mut buf = Vec::<u8>::with_capacity(text.len() + 1);
        buf.extend_from_slice(text.as_bytes());
        buf.push(0);

        let mut p: *const c_char = buf.as_ptr() as *const c_char;
        let text_mode: c_int = 1;
        let phonemes_mode: c_int = if self.tie {
            0x02 | (0x01 << 7) | ((0x0361u32 as c_int) << 8)
        } else {
            (((b'_' as u32) as c_int) << 8) | 0x02
        };

        let mut out = String::new();
        loop {
            if p.is_null() {
                break;
            }
            let first = unsafe { *p };
            if first == 0 {
                break;
            }

            let mut textptr: *const c_void = p as *const c_void;
            let phon = unsafe { espeak_TextToPhonemes(&mut textptr as *mut *const c_void, text_mode, phonemes_mode) };
            p = textptr as *const c_char;

            if phon.is_null() {
                break;
            }

            let s = unsafe { CStr::from_ptr(phon) }.to_str().unwrap_or("");
            out.push_str(s);
            if !p.is_null() && unsafe { *p } != 0 {
                out.push(' ');
            }
        }

        Ok(out)
    }
}

