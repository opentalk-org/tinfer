use std::collections::HashMap;
use std::str::FromStr;

use axum::Json;
use axum::body::Bytes;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use serde::{Deserialize, Serialize};

use crate::audio::{AudioEncoder, AudioFormat};
use crate::{AudioChunk, Error, ModelInfo};

#[derive(Deserialize)]
pub(crate) struct Speech {
    pub text: String,
    pub model_id: Option<String>,
    pub language_code: Option<String>,
}

#[derive(Deserialize)]
pub(crate) struct WsSpeech {
    pub text: String,
    #[serde(default)]
    pub try_trigger_generation: bool,
    #[serde(default)]
    pub flush: bool,
}

#[derive(Serialize)]
pub(crate) struct HealthResponse {
    pub ready: bool,
    pub status: &'static str,
}

#[derive(Serialize)]
pub(crate) struct LiveResponse {
    pub live: bool,
}

#[derive(Serialize)]
pub(crate) struct LanguageResponse {
    language_id: String,
    name: String,
}

#[derive(Serialize)]
pub(crate) struct ModelResponse {
    model_id: String,
    name: String,
    can_do_text_to_speech: bool,
    languages: Vec<LanguageResponse>,
    default_language: String,
}

#[derive(Serialize)]
pub(crate) struct VoiceResponse {
    voice_id: String,
    name: String,
    category: &'static str,
    labels: HashMap<String, String>,
    model_id: String,
}

impl VoiceResponse {
    pub(crate) fn new(voice_id: String, model_id: String) -> Self {
        Self { name: voice_id.clone(), voice_id, category: "generated", labels: HashMap::new(), model_id }
    }
}

#[derive(Serialize)]
pub(crate) struct VoicesResponse {
    pub voices: Vec<VoiceResponse>,
}

#[derive(Clone, Serialize)]
pub(crate) struct Timing {
    pub characters: Vec<String>,
    pub character_start_times_seconds: Vec<f64>,
    pub character_end_times_seconds: Vec<f64>,
}

#[derive(Serialize)]
pub(crate) struct TimedAudio {
    pub audio_base64: String,
    pub alignment: Timing,
    pub normalized_alignment: Timing,
}

impl From<crate::Alignment> for Timing {
    fn from(alignment: crate::Alignment) -> Self {
        Self {
            characters: alignment.items.iter().map(|item| item.item.clone()).collect(),
            character_start_times_seconds: alignment.items.iter().map(|item| item.start_ms as f64 / 1000.0).collect(),
            character_end_times_seconds: alignment.items.iter().map(|item| item.end_ms as f64 / 1000.0).collect(),
        }
    }
}

#[derive(Serialize)]
pub(crate) struct WsAudio {
    pub audio: String,
    #[serde(rename = "isFinal")]
    pub is_final: bool,
}

impl WsAudio {
    pub(crate) fn audio(bytes: Bytes) -> Self {
        use base64::Engine as _;
        Self { audio: base64::engine::general_purpose::STANDARD.encode(bytes), is_final: false }
    }

    pub(crate) fn final_message() -> Self {
        Self { audio: String::new(), is_final: true }
    }
}

#[derive(Serialize)]
struct ErrorBody {
    detail: ErrorDetail,
}

#[derive(Serialize)]
struct ErrorDetail {
    status: u16,
    message: String,
}

pub(crate) enum WebError {
    Validation(String),
    Engine(Error),
    Audio(crate::audio::AudioError),
    Unavailable,
}

impl From<Error> for WebError {
    fn from(error: Error) -> Self {
        Self::Engine(error)
    }
}

impl IntoResponse for WebError {
    fn into_response(self) -> Response {
        let (status, message) = match self {
            Self::Validation(message) => (StatusCode::UNPROCESSABLE_ENTITY, message),
            Self::Engine(Error::Catalog(message)) => (StatusCode::NOT_FOUND, message),
            Self::Engine(error) => (StatusCode::INTERNAL_SERVER_ERROR, error.to_string()),
            Self::Audio(error) => (StatusCode::INTERNAL_SERVER_ERROR, error.to_string()),
            Self::Unavailable => (StatusCode::SERVICE_UNAVAILABLE, "server is not accepting synthesis requests".into()),
        };
        let body = ErrorBody { detail: ErrorDetail { status: status.as_u16(), message } };
        (status, Json(body)).into_response()
    }
}

pub(crate) fn output_format(query: &HashMap<String, String>) -> Result<AudioFormat, WebError> {
    AudioFormat::from_str(query.get("output_format").map_or("mp3_44100_128", String::as_str))
        .map_err(|error| WebError::Validation(error.to_string()))
}

pub(crate) fn encode(chunk: AudioChunk, format: AudioFormat) -> Result<Bytes, WebError> {
    let mut encoder = AudioEncoder::new(format, chunk.sample_rate).map_err(WebError::Audio)?;
    let mut output = encoder.push(&chunk.audio).map_err(WebError::Audio)?.to_vec();
    output.extend_from_slice(&encoder.finish().map_err(WebError::Audio)?);
    Ok(Bytes::from(output))
}

pub(crate) fn format_model(info: ModelInfo) -> ModelResponse {
    let languages =
        info.supported_languages.into_iter().map(|language_id| LanguageResponse { name: language_id.clone(), language_id }).collect();
    ModelResponse {
        name: info.model_id.clone(),
        model_id: info.model_id,
        can_do_text_to_speech: true,
        languages,
        default_language: info.default_language,
    }
}
