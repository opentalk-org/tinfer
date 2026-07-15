mod multi;
mod server;
mod single;
pub(super) mod wire;

pub use server::{WebConfig, WebServer};

use std::collections::HashMap;
use std::sync::Arc;

use axum::Json;
use axum::Router;
use axum::extract::{Path, Query, State};
use axum::http::header;
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use base64::Engine as _;

use self::wire::{
    HealthResponse, LiveResponse, ModelResponse, TimedAudio, Timing, VoiceResponse, VoicesResponse, WebError, encode, format_model,
};
use super::health::HealthState;
use super::http::{Speech, Transport, parse_query, parse_speech, speech_stream, timing_stream};
use crate::audio::AudioEncoding;
use crate::{Alignment, AlignmentType, AsyncEngine, AudioChunk, Error};

#[derive(Clone)]
pub(super) struct App {
    pub(super) engine: AsyncEngine,
    pub(super) health: Arc<HealthState>,
}

pub(super) fn router(engine: AsyncEngine, health: Arc<HealthState>) -> Router {
    let app = App { engine, health };
    Router::new()
        .route("/health", get(health_status))
        .route("/health/live", get(live))
        .route("/health/ready", get(ready))
        .route("/livez", get(live))
        .route("/readyz", get(ready))
        .route("/v1/models", get(models))
        .route("/v1/voices", get(voices))
        .route("/v1/text-to-speech/{voice}/stream/with-timestamps", post(timing_stream))
        .route("/v1/text-to-speech/{voice}/with-timestamps", post(timing))
        .route("/v1/text-to-speech/{voice}/stream", post(speech_stream))
        .route("/v1/text-to-speech/{voice}", post(speech))
        .route("/v1/text-to-speech/{voice}/stream-input", get(single::upgrade))
        .route("/v1/text-to-speech/{voice}/multi-stream-input", get(multi::upgrade))
        .with_state(app)
}

async fn health_status(State(app): State<App>) -> Response {
    let ready = app.health.ready();
    let status = if ready { axum::http::StatusCode::OK } else { axum::http::StatusCode::SERVICE_UNAVAILABLE };
    (status, Json(HealthResponse { ready, status: if ready { "serving" } else { "not_serving" } })).into_response()
}

async fn live(State(app): State<App>) -> Response {
    let live = app.health.state() != super::ServingState::Stopped;
    let status = if live { axum::http::StatusCode::OK } else { axum::http::StatusCode::SERVICE_UNAVAILABLE };
    (status, Json(LiveResponse { live, status: if live { "serving" } else { "not_serving" } })).into_response()
}

async fn ready(State(app): State<App>) -> Response {
    let ready = app.health.ready();
    let status = if ready { axum::http::StatusCode::OK } else { axum::http::StatusCode::SERVICE_UNAVAILABLE };
    (status, Json(HealthResponse { ready, status: if ready { "serving" } else { "not_serving" } })).into_response()
}

async fn models(State(app): State<App>) -> Result<Json<Vec<ModelResponse>>, WebError> {
    Ok(Json(app.engine.get_model_infos().await?.into_iter().map(format_model).collect()))
}

async fn voices(State(app): State<App>, Query(query): Query<HashMap<String, String>>) -> Result<Json<VoicesResponse>, WebError> {
    let models = match query.get("model_id") {
        Some(model) => vec![model.clone()],
        None => app.engine.get_model_ids().await?,
    };
    let mut voices = Vec::new();
    for model in models {
        for voice in app.engine.get_voice_ids(&model).await? {
            voices.push(VoiceResponse::new(voice, model.clone()));
        }
    }
    Ok(Json(VoicesResponse { voices }))
}

async fn speech(
    State(app): State<App>,
    Path(voice): Path<String>,
    Query(query): Query<HashMap<String, String>>,
    Json(value): Json<serde_json::Value>,
) -> Result<Response, WebError> {
    let request = parse_speech(value)?;
    let _admission = app.health.admit().ok_or(WebError::Unavailable)?;
    let format = parse_query(&query, Transport::Http)?.output_format;
    let content_type = match format.encoding {
        AudioEncoding::Mp3 => "audio/mpeg",
        AudioEncoding::Opus => "audio/ogg",
        AudioEncoding::WavPcm16 => "audio/wav",
        _ => "application/octet-stream",
    };
    let chunk = generate(&app.engine, voice, request).await?;
    Ok(([(header::CONTENT_TYPE, content_type)], encode(chunk, format)?).into_response())
}

async fn timing(
    State(app): State<App>,
    Path(voice): Path<String>,
    Query(query): Query<HashMap<String, String>>,
    Json(value): Json<serde_json::Value>,
) -> Result<Json<TimedAudio>, WebError> {
    let request = parse_speech(value)?;
    let _admission = app.health.admit().ok_or(WebError::Unavailable)?;
    let format = parse_query(&query, Transport::Http)?.output_format;
    let chunk = generate_timed(&app.engine, voice, request).await?;
    let timing = Timing::from(chunk.alignment.clone().unwrap_or_default());
    Ok(Json(TimedAudio {
        audio_base64: base64::engine::general_purpose::STANDARD.encode(encode(chunk, format)?),
        alignment: timing.clone(),
        normalized_alignment: timing,
    }))
}

async fn generate(engine: &AsyncEngine, voice: String, mut request: Speech) -> Result<AudioChunk, WebError> {
    if request.text.trim().is_empty() {
        return Err(WebError::Validation("text must contain speech content".into()));
    }
    let model = resolve_model(engine, request.model_id.clone()).await?;
    request.language_code = Some(resolve_language(engine, &model, request.language_code.as_deref()).await?);
    let params = request.stream_params(AlignmentType::None);
    Ok(engine.generate_full(&model, &voice, &request.text, params).await?)
}

async fn generate_timed(engine: &AsyncEngine, voice: String, mut request: Speech) -> Result<AudioChunk, WebError> {
    if request.text.trim().is_empty() {
        return Err(WebError::Validation("text must contain speech content".into()));
    }
    let model = resolve_model(engine, request.model_id.clone()).await?;
    request.language_code = Some(resolve_language(engine, &model, request.language_code.as_deref()).await?);
    let params = request.stream_params(AlignmentType::Char);
    let stream = engine.create_stream(&model, &voice, params).await?;
    stream.add_text(&request.text).await?;
    stream.force_generate().await?;
    let mut chunks = Vec::new();
    while let Some(chunk) = stream.recv().await? {
        chunks.push(chunk);
    }
    stream.close().await?;
    merge_timed(chunks)
}

fn merge_timed(chunks: Vec<AudioChunk>) -> Result<AudioChunk, WebError> {
    let alignments = chunks.iter().filter_map(|chunk| chunk.alignment.as_ref()).flat_map(|alignment| alignment.items.clone()).collect();
    let mut merged = AudioChunk::merge(chunks)?;
    merged.alignment = Some(Alignment { items: alignments, kind: AlignmentType::Char });
    Ok(merged)
}

pub(super) async fn resolve_model(engine: &AsyncEngine, requested: Option<String>) -> Result<String, Error> {
    let models = engine.get_model_ids().await?;
    match requested {
        Some(model) if models.contains(&model) => Ok(model),
        Some(model) => Err(Error::Catalog(format!("unknown model: {model}"))),
        None => models.into_iter().next().ok_or_else(|| Error::Catalog("no models loaded".into())),
    }
}

pub(super) async fn resolve_language(engine: &AsyncEngine, model: &str, requested: Option<&str>) -> Result<String, Error> {
    let info = engine
        .get_model_infos()
        .await?
        .into_iter()
        .find(|info| info.model_id == model)
        .ok_or_else(|| Error::Catalog(format!("unknown model: {model}")))?;
    Ok(requested
        .filter(|language| info.supported_languages.iter().any(|supported| supported == language))
        .unwrap_or(&info.default_language)
        .to_owned())
}
