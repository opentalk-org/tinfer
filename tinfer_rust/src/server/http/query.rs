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
}

pub(crate) fn parse_query(values: &HashMap<String, String>, transport: Transport) -> Result<SpeechQuery, WebError> {
    let allowed = match transport {
        Transport::Http => &["output_format", "enable_logging", "optimize_streaming_latency"][..],
        Transport::WebSocket => &[
            "model_id", "output_format", "language_code", "sync_alignment", "inactivity_timeout", "auto_mode", "enable_logging",
            "enable_ssml_parsing", "apply_text_normalization", "seed", "authorization", "single_use_token",
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
    if let Some(mode) = values.get("apply_text_normalization") {
        if !matches!(mode.as_str(), "auto" | "on" | "off") {
            return Err(WebError::Validation("apply_text_normalization must be auto, on, or off".into()));
        }
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
    Ok(SpeechQuery { output_format: format, inactivity_timeout: Duration::from_secs(timeout) })
}

fn parse_bool(values: &HashMap<String, String>, field: &str) -> Result<(), WebError> {
    if let Some(value) = values.get(field) {
        if !matches!(value.to_ascii_lowercase().as_str(), "true" | "false") {
            return Err(WebError::Validation(format!("{field} must be true or false")));
        }
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
