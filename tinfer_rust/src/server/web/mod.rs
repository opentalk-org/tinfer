mod server;
pub(super) mod wire;

pub use server::{WebConfig, WebServer};

use std::collections::HashMap;
use std::sync::Arc;

use axum::Json;
use axum::Router;
use axum::extract::ws::{CloseFrame, Message, WebSocket, WebSocketUpgrade};
use axum::extract::{Path, Query, State};
use axum::http::header;
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use base64::Engine as _;
use futures_util::StreamExt;

use self::wire::{
    HealthResponse, LiveResponse, ModelResponse, TimedAudio, Timing, VoiceResponse, VoicesResponse, WebError, WsAudio, WsSpeech, encode,
    format_model,
};
use super::http::{Speech, Transport, parse_query, parse_speech, speech_stream, timing_stream};
use super::health::HealthState;
use crate::audio::{AudioEncoding, AudioFormat};
use crate::{Alignment, AlignmentType, AsyncEngine, AudioChunk, Error, StreamParams};

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

async fn websocket(
    ws: WebSocketUpgrade,
    State(app): State<App>,
    Path(voice): Path<String>,
    Query(query): Query<HashMap<String, String>>,
) -> Result<Response, WebError> {
    let admission = app.health.admit().ok_or(WebError::Unavailable)?;
    let parsed = parse_query(&query, Transport::WebSocket)?;
    let format = parsed.output_format;
    let model = resolve_model(&app.engine, query.get("model_id").cloned()).await?;
    if !app.engine.get_voice_ids(&model).await?.contains(&voice) {
        return Err(Error::Catalog(format!("unknown voice: {voice}")).into());
    }
    Ok(ws.on_upgrade(move |socket| websocket_session(socket, app.engine, model, voice, format, parsed, admission)))
}

async fn websocket_session(
    mut socket: WebSocket,
    engine: AsyncEngine,
    model: String,
    voice: String,
    format: AudioFormat,
    query: super::http::SpeechQuery,
    _admission: super::health::Admission,
) {
    let params = StreamParams {
        alignment_type: if query.sync_alignment { AlignmentType::Char } else { AlignmentType::None },
        ..StreamParams::default()
    };
    let Ok(stream) = engine.create_stream(&model, &voice, params).await else {
        return;
    };
    let (audio_tx, mut audio_rx) = tokio::sync::mpsc::channel(4);
    let audio_stream = stream.clone();
    let pump = tokio::spawn(async move {
        loop {
            let delivery = audio_stream.recv_marked().await;
            let finished = matches!(delivery, Ok((None, true)) | Err(_));
            if audio_tx.send(delivery).await.is_err() || finished {
                break;
            }
        }
    });
    let mut initialized = false;
    let mut finalized = false;
    let deadline = tokio::time::sleep(query.inactivity_timeout);
    tokio::pin!(deadline);
    loop {
        tokio::select! {
            frame = socket.next() => match frame {
                Some(Ok(Message::Text(raw))) => {
                    deadline.as_mut().reset(tokio::time::Instant::now() + query.inactivity_timeout);
                    let Ok(message) = serde_json::from_str::<WsSpeech>(&raw) else { policy_close(&mut socket, "message must be valid JSON").await; break };
                    if !initialized {
                        if message.text != " " { policy_close(&mut socket, "first message text must be exactly \" \"").await; break }
                        initialized = true;
                    } else if message.text.is_empty() {
                        finalized = stream.finish().await.is_ok();
                    } else if !message.text.ends_with(' ') {
                        policy_close(&mut socket, "text must end in a space").await; break
                    } else {
                        if stream.add_text(&message.text).await.is_err() { break }
                        if message.flush { let _ = stream.force_generate().await; }
                        else if message.try_trigger_generation { let _ = stream.try_generate().await; }
                    }
                }
                Some(Ok(Message::Binary(_))) => { policy_close(&mut socket, "binary messages are not supported").await; break }
                Some(Ok(Message::Close(_))) | None | Some(Err(_)) => break,
                _ => {}
            },
            delivery = audio_rx.recv(), if initialized => match delivery {
                Some(Ok((Some(chunk), _))) => {
                    let alignment = if query.sync_alignment { chunk.alignment.clone() } else { None };
                    let Ok(bytes) = encode(chunk, format) else { break };
                    let payload = serde_json::to_string(&WsAudio::audio(bytes, alignment)).expect("serializable response");
                    if socket.send(Message::Text(payload.into())).await.is_err() { break }
                }
                Some(Ok((None, true))) => {
                    let payload = serde_json::to_string(&WsAudio::final_message()).expect("serializable response");
                    let _ = socket.send(Message::Text(payload.into())).await;
                    let _ = socket.send(Message::Close(Some(CloseFrame { code: 1000, reason: "complete".into() }))).await;
                    break;
                }
                Some(Err(_)) | None => break,
                Some(Ok((None, false))) => {}
            },
            () = &mut deadline, if initialized && !finalized => {
                policy_close(&mut socket, "inactivity timeout").await; break
            }
        }
    }
    if !finalized {
        let _ = stream.cancel().await;
    }
    pump.abort();
    let _ = stream.close().await;
}

async fn policy_close(socket: &mut WebSocket, error: &str) {
    let payload = serde_json::json!({"error": error}).to_string();
    let _ = socket.send(Message::Text(payload.into())).await;
    let _ = socket.send(Message::Close(Some(CloseFrame { code: 1008, reason: error.into() }))).await;
}

async fn generate(engine: &AsyncEngine, voice: String, request: Speech) -> Result<AudioChunk, WebError> {
    if request.text.trim().is_empty() {
        return Err(WebError::Validation("text must contain speech content".into()));
    }
    let model = resolve_model(engine, request.model_id).await?;
    let params = StreamParams { model: serde_json::json!({ "language": request.language_code }), ..StreamParams::default() };
    Ok(engine.generate_full(&model, &voice, &request.text, params).await?)
}

async fn generate_timed(engine: &AsyncEngine, voice: String, request: Speech) -> Result<AudioChunk, WebError> {
    if request.text.trim().is_empty() {
        return Err(WebError::Validation("text must contain speech content".into()));
    }
    let model = resolve_model(engine, request.model_id).await?;
    let params = StreamParams {
        alignment_type: AlignmentType::Char,
        model: serde_json::json!({ "language": request.language_code }),
        ..StreamParams::default()
    };
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
