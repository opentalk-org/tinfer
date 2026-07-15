use std::collections::HashMap;
use std::str::FromStr;
use std::time::Duration;

use crate::audio::{AudioEncoding, AudioFormat};
use crate::server::web::wire::WebError;

#[derive(Clone, Copy, Eq, PartialEq)]
pub(crate) enum Transport {
    Http,
    WebSocket,
}

pub(crate) struct SpeechQuery {
    pub output_format: AudioFormat,
    pub inactivity_timeout: Duration,
    pub sync_alignment: bool,
    pub language_code: Option<String>,
    pub auto_mode: bool,
    pub seed: Option<u64>,
    pub apply_text_normalization: String,
}

pub(crate) fn parse_query(values: &HashMap<String, String>, transport: Transport) -> Result<SpeechQuery, WebError> {
    let allowed = match transport {
        Transport::Http => &["output_format", "enable_logging", "optimize_streaming_latency"][..],
        Transport::WebSocket => &[
            "model_id",
            "output_format",
            "language_code",
            "sync_alignment",
            "inactivity_timeout",
            "auto_mode",
            "enable_logging",
            "enable_ssml_parsing",
            "apply_text_normalization",
            "seed",
            "authorization",
            "single_use_token",
        ][..],
    };
    if let Some(field) = values.keys().filter(|key| !allowed.contains(&key.as_str())).min() {
        return Err(WebError::Validation(format!("unsupported query parameter: {field}")));
    }
    parse_bool(values, "enable_logging")?;
    if transport == Transport::WebSocket {
        for field in ["sync_alignment", "auto_mode", "enable_ssml_parsing"] {
            parse_bool(values, field)?;
        }
        bounded_u64(values, "seed", 0, u32::MAX as u64)?;
    } else {
        bounded_u64(values, "optimize_streaming_latency", 0, 4)?;
    }
    if values.get("apply_text_normalization").is_some_and(|mode| !matches!(mode.as_str(), "auto" | "on" | "off")) {
        return Err(WebError::Validation("apply_text_normalization must be auto, on, or off".into()));
    }
    let timeout = bounded_u64(values, "inactivity_timeout", 1, 180)?.unwrap_or(20);
    let format = AudioFormat::from_str(values.get("output_format").map_or("mp3_44100_128", String::as_str))
        .map_err(|error| WebError::Validation(error.to_string()))?;
    if transport == Transport::WebSocket
        && (format.encoding == AudioEncoding::WavPcm16
            || matches!((format.encoding, format.sample_rate), (AudioEncoding::Pcm16, 32_000 | 48_000))
            || matches!((format.encoding, format.sample_rate, format.bitrate_kbps), (AudioEncoding::Mp3, 24_000, Some(48))))
    {
        return Err(WebError::Validation("unsupported output_format for WebSocket".into()));
    }
    let sync_alignment = values.get("sync_alignment").is_some_and(|value| value.eq_ignore_ascii_case("true"));
    let language_code = optional_string(values, "language_code")?;
    let auto_mode = values.get("auto_mode").is_some_and(|value| value.eq_ignore_ascii_case("true"));
    let seed = bounded_u64(values, "seed", 0, u32::MAX as u64)?;
    let apply_text_normalization = values.get("apply_text_normalization").cloned().unwrap_or_else(|| "auto".into());
    Ok(SpeechQuery {
        output_format: format,
        inactivity_timeout: Duration::from_secs(timeout),
        sync_alignment,
        language_code,
        auto_mode,
        seed,
        apply_text_normalization,
    })
}

impl SpeechQuery {
    pub(crate) fn stream_params(&self, speech: &super::Speech, alignment_type: crate::AlignmentType) -> crate::StreamParams {
        let mut params = speech.stream_params(alignment_type);
        params.timeout = if self.auto_mode { Duration::ZERO } else { Duration::from_millis(80) };
        let model = params.model.as_object_mut().expect("speech parameters are an object");
        if let (None, Some(seed)) = (speech.seed, self.seed) {
            model.insert("seed".into(), serde_json::json!(seed));
        }
        if speech.apply_text_normalization.is_none() {
            model.insert("apply_text_normalization".into(), serde_json::json!(self.apply_text_normalization));
        }
        if let Some(language) = &self.language_code {
            model.insert("language".into(), serde_json::json!(language));
        }
        params
    }
}

fn parse_bool(values: &HashMap<String, String>, field: &str) -> Result<(), WebError> {
    if values.get(field).is_some_and(|value| !matches!(value.to_ascii_lowercase().as_str(), "true" | "false")) {
        return Err(WebError::Validation(format!("{field} must be true or false")));
    }
    Ok(())
}

fn bounded_u64(values: &HashMap<String, String>, field: &str, minimum: u64, maximum: u64) -> Result<Option<u64>, WebError> {
    let Some(raw) = values.get(field) else { return Ok(None) };
    let value = raw.parse::<u64>().map_err(|_| WebError::Validation(format!("{field} must be an integer")))?;
    if !(minimum..=maximum).contains(&value) {
        return Err(WebError::Validation(format!("{field} must be between {minimum} and {maximum}")));
    }
    Ok(Some(value))
}

fn optional_string(values: &HashMap<String, String>, field: &str) -> Result<Option<String>, WebError> {
    match values.get(field) {
        None => Ok(None),
        Some(value) if !value.is_empty() => Ok(Some(value.clone())),
        Some(_) => Err(WebError::Validation(format!("{field} must be a non-empty string"))),
    }
}
