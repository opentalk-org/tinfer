mod server;
mod wire;

pub use server::{WebConfig, WebServer};

use std::collections::HashMap;
use std::sync::Arc;

use axum::Json;
use axum::Router;
use axum::extract::ws::{Message, WebSocket, WebSocketUpgrade};
use axum::extract::{Path, Query, State};
use axum::http::header;
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use base64::Engine as _;
use futures_util::{SinkExt, StreamExt};

use self::wire::{
    HealthResponse, LiveResponse, ModelResponse, Speech, TimedAudio, Timing, VoiceResponse, VoicesResponse, WebError, WsAudio, WsSpeech,
    encode, format_model, output_format,
};
use super::health::HealthState;
use crate::audio::{AudioEncoding, AudioFormat};
use crate::{AsyncEngine, AudioChunk, Error, StreamParams};

#[derive(Clone)]
struct App {
    engine: AsyncEngine,
    health: Arc<HealthState>,
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
        .route("/v1/text-to-speech/{voice}/stream/with-timestamps", post(timing))
        .route("/v1/text-to-speech/{voice}/with-timestamps", post(timing))
        .route("/v1/text-to-speech/{voice}/stream", post(speech))
        .route("/v1/text-to-speech/{voice}", post(speech))
        .route("/v1/text-to-speech/{voice}/stream-input", get(websocket))
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
    (status, Json(LiveResponse { live })).into_response()
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
    Json(request): Json<Speech>,
) -> Result<Response, WebError> {
    let _admission = app.health.admit().ok_or(WebError::Unavailable)?;
    let format = output_format(&query)?;
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
    Json(request): Json<Speech>,
) -> Result<Json<TimedAudio>, WebError> {
    let _admission = app.health.admit().ok_or(WebError::Unavailable)?;
    let format = output_format(&query)?;
    let chunk = generate(&app.engine, voice, request).await?;
    let timing = Timing::from(chunk.alignment.clone().unwrap_or_default());
    Ok(Json(TimedAudio {
        audio_base64: base64::engine::general_purpose::STANDARD.encode(encode(chunk, format)?),
        alignment: timing.clone(),
        normalized_alignment: timing,
    }))
}

async fn websocket(
    ws: WebSocketUpgrade,
    State(app): State<App>,
    Path(voice): Path<String>,
    Query(query): Query<HashMap<String, String>>,
) -> Result<Response, WebError> {
    let admission = app.health.admit().ok_or(WebError::Unavailable)?;
    let format = output_format(&query)?;
    let model = resolve_model(&app.engine, query.get("model_id").cloned()).await?;
    if !app.engine.get_voice_ids(&model).await?.contains(&voice) {
        return Err(Error::Catalog(format!("unknown voice: {voice}")).into());
    }
    Ok(ws.on_upgrade(move |socket| websocket_session(socket, app.engine, model, voice, format, admission)))
}

async fn websocket_session(
    socket: WebSocket,
    engine: AsyncEngine,
    model: String,
    voice: String,
    format: AudioFormat,
    _admission: super::health::Admission,
) {
    let Ok(stream) = engine.create_stream(&model, &voice, StreamParams::default()).await else {
        return;
    };
    let (mut output, mut input) = socket.split();
    let output_stream = stream.clone();
    let writer = tokio::spawn(async move {
        loop {
            match output_stream.recv_marked().await {
                Ok((Some(chunk), _)) => {
                    let Ok(audio) = encode(chunk, format) else {
                        break;
                    };
                    let message = serde_json::to_string(&WsAudio::audio(audio)).expect("WebSocket response is serializable");
                    if output.send(Message::Text(message.into())).await.is_err() {
                        break;
                    }
                }
                Ok((None, true)) => {
                    let message = serde_json::to_string(&WsAudio::final_message()).expect("WebSocket response is serializable");
                    let _ = output.send(Message::Text(message.into())).await;
                    break;
                }
                Ok((None, false)) => {}
                Err(_) => break,
            }
        }
    });
    let mut initialized = false;
    let mut finalized = false;
    while let Some(Ok(Message::Text(raw))) = input.next().await {
        let Ok(message) = serde_json::from_str::<WsSpeech>(&raw) else {
            break;
        };
        if !initialized {
            if message.text != " " {
                break;
            }
            initialized = true;
            continue;
        }
        if message.text.is_empty() {
            finalized = stream.finish().await.is_ok();
            break;
        }
        if stream.add_text(&message.text).await.is_err() {
            break;
        }
        if message.flush && stream.force_generate().await.is_err() {
            break;
        }
        if !message.flush && message.try_trigger_generation && stream.try_generate().await.is_err() {
            break;
        }
    }
    if !finalized {
        let _ = stream.cancel().await;
    }
    let _ = writer.await;
    let _ = stream.close().await;
}

async fn generate(engine: &AsyncEngine, voice: String, request: Speech) -> Result<AudioChunk, WebError> {
    if request.text.trim().is_empty() {
        return Err(WebError::Validation("text must contain speech content".into()));
    }
    let model = resolve_model(engine, request.model_id).await?;
    let params = StreamParams { model: serde_json::json!({ "language": request.language_code }), ..StreamParams::default() };
    Ok(engine.generate_full(&model, &voice, &request.text, params).await?)
}

async fn resolve_model(engine: &AsyncEngine, requested: Option<String>) -> Result<String, Error> {
    let models = engine.get_model_ids().await?;
    match requested {
        Some(model) if models.contains(&model) => Ok(model),
        Some(model) => Err(Error::Catalog(format!("unknown model: {model}"))),
        None => models.into_iter().next().ok_or_else(|| Error::Catalog("no models loaded".into())),
    }
}
