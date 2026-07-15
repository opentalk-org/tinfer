use std::collections::HashMap;
use std::sync::Mutex;

use espeak_align_core::Engine;
use serde::Deserialize;

use super::manifest::Manifest;
use super::text::{PreparedText, prepare_texts, tokenize};
use crate::models::base::native::{ffi, tensor};
use crate::{Error, ModelRequest, Result};

const MAX_DIFFUSION_STEPS: usize = 5;

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct StyleTts2Params {
    pub use_diffusion: bool,
    pub phonemized: bool,
    pub language: String,
    pub embedding_scale: f32,
    pub diffusion_steps: usize,
    pub style_interpolation_factor: f32,
    pub alpha: f32,
    pub beta: f32,
    pub speed: f32,
    pub seed: Option<u64>,
    pub apply_text_normalization: String,
}

pub(super) struct PreparedBatch {
    pub tensors: Vec<ffi::Tensor>,
    pub texts: Vec<PreparedText>,
}

pub(super) fn validate_settings(value: &serde_json::Value) -> Result<()> {
    if !value.as_object().is_some_and(|settings| settings.contains_key("seed")) {
        return Err(Error::Validation("StyleTTS2 setting seed is required; use null for no fixed seed".into()));
    }
    serde_json::from_value::<StyleTts2Params>(value.clone())
        .map(|_| ())
        .map_err(|error| Error::Validation(format!("invalid StyleTTS2 settings: {error}")))
}

pub(super) fn prepare(
    requests: &[ModelRequest],
    manifest: &Manifest,
    voices: &Mutex<HashMap<String, Vec<f32>>>,
    phonemizers: &Mutex<HashMap<String, Engine>>,
) -> Result<PreparedBatch> {
    let mut params = Vec::with_capacity(requests.len());
    for request in requests {
        let mut value = parameters(request)?;
        if value.language.is_empty() {
            value.language.clone_from(&manifest.default_language);
        }
        if !manifest.supported_languages.contains(&value.language)
            || value.diffusion_steps < 2
            || value.diffusion_steps > MAX_DIFFUSION_STEPS
            || !value.speed.is_finite()
            || value.speed <= 0.0
            || !value.alpha.is_finite()
            || !value.beta.is_finite()
            || !value.embedding_scale.is_finite()
            || !matches!(value.apply_text_normalization.as_str(), "auto" | "on" | "off")
        {
            return Err(Error::Validation("invalid StyleTTS2 generation parameters".into()));
        }
        params.push(value);
    }
    let request_texts = requests.iter().map(|request| request.text.as_str()).collect::<Vec<_>>();
    let phonemized = params.iter().map(|value| value.phonemized).collect::<Vec<_>>();
    let languages = params.iter().map(|value| value.language.as_str()).collect::<Vec<_>>();
    let normalization = params.iter().map(|value| value.apply_text_normalization.as_str()).collect::<Vec<_>>();
    let texts = prepare_texts(&request_texts, &phonemized, &languages, &normalization, &manifest.symbols, phonemizers)?;
    let all_tokens = texts.iter().map(|text| tokenize(&text.phonemes, &manifest.symbols)).collect::<Vec<_>>();
    if all_tokens.iter().any(|tokens| tokens.len() < 2 || tokens.len() > 512) {
        return Err(Error::Validation("StyleTTS2 text must contain 1..511 model tokens".into()));
    }
    let mut voice_vectors = Vec::with_capacity(requests.len() * 256);
    for request in requests {
        let mut cache = voices.lock().expect("StyleTTS2 voice cache lock poisoned");
        if !cache.contains_key(&request.voice_id) {
            cache.insert(request.voice_id.clone(), manifest.load_voice(&request.voice_id)?);
        }
        voice_vectors.extend_from_slice(&cache[&request.voice_id]);
    }
    let batch = requests.len();
    let width = all_tokens.iter().map(Vec::len).max().expect("model batch is non-empty");
    let mut padded = vec![0_i64; batch * width];
    let mut mask = vec![1_u8; batch * width];
    let mut lengths = Vec::with_capacity(batch);
    for (item, tokens) in all_tokens.iter().enumerate() {
        padded[item * width..item * width + tokens.len()].copy_from_slice(tokens);
        mask[item * width..item * width + tokens.len()].fill(0);
        lengths.push(tokens.len() as i32);
    }
    let seeds = params.iter().enumerate().map(|(index, value)| value.seed.unwrap_or(index as u64)).collect::<Vec<_>>();
    let noise = seeds.iter().flat_map(|seed| gaussian(*seed, 256)).collect::<Vec<_>>();
    let step_noise = seeds.iter().flat_map(|seed| gaussian(seed.wrapping_add(1), 4 * 256)).collect::<Vec<_>>();
    let speeds = params
        .iter()
        .zip(&all_tokens)
        .map(|(value, tokens)| value.speed * (1.4 - 0.00115 * tokens.len().saturating_sub(1).clamp(1, 300) as f32))
        .collect::<Vec<_>>();
    let alpha = params.iter().map(|value| value.alpha).collect::<Vec<_>>();
    let beta = params.iter().map(|value| value.beta).collect::<Vec<_>>();
    let scales = params.iter().map(|value| value.embedding_scale).collect::<Vec<_>>();
    let diffusion = params.iter().map(|value| u8::from(value.use_diffusion)).collect::<Vec<_>>();
    let interpolation = params.iter().map(|value| value.style_interpolation_factor).collect::<Vec<_>>();
    let mut previous = Vec::with_capacity(batch * 256);
    let mut has_previous = Vec::with_capacity(batch);
    for request in requests {
        let state = request.state["previous_style_vector"].as_array();
        has_previous.push(u8::from(state.is_some()));
        if let Some(values) = state {
            if values.len() != 256 {
                return Err(Error::Validation("StyleTTS2 previous style must contain 256 floats".into()));
            }
            previous.extend(
                values
                    .iter()
                    .map(|value| value.as_f64().ok_or_else(|| Error::Validation("invalid StyleTTS2 previous style".into())))
                    .collect::<Result<Vec<_>>>()?
                    .into_iter()
                    .map(|value| value as f32),
            );
        } else {
            previous.extend(std::iter::repeat_n(0.0, 256));
        }
    }
    let sigmas = diffusion_schedule(params[0].diffusion_steps);
    let tensors = vec![
        tensor("operation", ffi::DType::I32, vec![1], bytes(&[0_i32])),
        tensor(
            "stream_ids",
            ffi::DType::I64,
            vec![batch as i64],
            bytes(&requests.iter().map(|request| request.stream_id as i64).collect::<Vec<_>>()),
        ),
        tensor("tokens", ffi::DType::I64, vec![batch as i64, width as i64], bytes(&padded)),
        tensor("mask", ffi::DType::Bool, vec![batch as i64, width as i64], mask),
        tensor("ref_s", ffi::DType::F32, vec![batch as i64, 256], bytes(&voice_vectors)),
        tensor("noise", ffi::DType::F32, vec![batch as i64, 1, 256], bytes(&noise)),
        tensor("step_noise", ffi::DType::F32, vec![batch as i64, 4, 1, 256], bytes(&step_noise)),
        tensor("alpha", ffi::DType::F32, vec![batch as i64, 1], bytes(&alpha)),
        tensor("beta", ffi::DType::F32, vec![batch as i64, 1], bytes(&beta)),
        tensor("scale", ffi::DType::F32, vec![batch as i64], bytes(&scales)),
        tensor("sigmas", ffi::DType::F32, vec![6], bytes(&sigmas)),
        tensor("use_diffusion", ffi::DType::Bool, vec![batch as i64], diffusion),
        tensor("previous_s", ffi::DType::F32, vec![batch as i64, 256], bytes(&previous)),
        tensor("has_previous", ffi::DType::Bool, vec![batch as i64], has_previous),
        tensor("style_interpolation", ffi::DType::F32, vec![batch as i64], bytes(&interpolation)),
        tensor("lengths", ffi::DType::I32, vec![batch as i64], bytes(&lengths)),
        tensor("speed", ffi::DType::F32, vec![batch as i64], bytes(&speeds)),
        tensor("seeds", ffi::DType::I64, vec![batch as i64], bytes(&seeds)),
    ];
    Ok(PreparedBatch { tensors, texts })
}

pub(super) fn parameters(request: &ModelRequest) -> Result<StyleTts2Params> {
    serde_json::from_value(request.params.clone()).map_err(|error| Error::Validation(format!("invalid StyleTTS2 parameters: {error}")))
}

pub(super) fn bytes<T: Copy>(values: &[T]) -> Vec<u8> {
    let length = std::mem::size_of_val(values);
    let pointer = values.as_ptr().cast::<u8>();
    unsafe { std::slice::from_raw_parts(pointer, length) }.to_vec()
}

fn gaussian(mut state: u64, count: usize) -> Vec<f32> {
    let mut output = Vec::with_capacity(count);
    while output.len() < count {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        let first = ((state >> 11) as f64 + 1.0) / ((1_u64 << 53) as f64 + 1.0);
        state = state.wrapping_mul(0x9E3779B97F4A7C15);
        let second = ((state >> 11) as f64 + 1.0) / ((1_u64 << 53) as f64 + 1.0);
        output.push((-2.0 * first.ln()).sqrt().mul_add((std::f64::consts::TAU * second).cos(), 0.0) as f32);
    }
    output
}

pub(super) fn diffusion_schedule(steps: usize) -> Vec<f32> {
    assert!((2..=MAX_DIFFUSION_STEPS).contains(&steps), "diffusion step invariant violated");
    let start = 3.0_f32.powf(1.0 / 9.0);
    let end = 0.0001_f32.powf(1.0 / 9.0);
    let mut schedule = (0..steps).map(|index| (start + (end - start) * index as f32 / (steps - 1) as f32).powi(9)).collect::<Vec<_>>();
    *schedule.last_mut().expect("validated diffusion schedule is non-empty") = 0.0001;
    schedule.resize(MAX_DIFFUSION_STEPS + 1, 0.0001);
    schedule
}
