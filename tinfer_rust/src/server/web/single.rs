use std::collections::HashMap;

use axum::extract::ws::{CloseFrame, Message, WebSocket, WebSocketUpgrade};
use axum::extract::{Path, Query, State};
use axum::response::Response;
use futures_util::StreamExt;
use serde_json::{Map, Value, json};
use tokio::sync::mpsc;

use super::wire::{WebError, WsAudio, encode};
use super::{App, resolve_language, resolve_model};
use crate::server::http::{Speech, SpeechQuery, Transport, parse_query, parse_speech};
use crate::{AlignmentType, AsyncStream, Error};

pub(super) async fn upgrade(
    ws: WebSocketUpgrade,
    State(app): State<App>,
    Path(voice): Path<String>,
    Query(values): Query<HashMap<String, String>>,
) -> Result<Response, WebError> {
    let mut query = parse_query(&values, Transport::WebSocket)?;
    let model = resolve_model(&app.engine, values.get("model_id").cloned()).await?;
    if !app.engine.get_voice_ids(&model).await?.contains(&voice) {
        return Err(Error::Catalog(format!("unknown voice: {voice}")).into());
    }
    query.language_code = Some(resolve_language(&app.engine, &model, query.language_code.as_deref()).await?);
    let admission = app.health.admit().ok_or(WebError::Unavailable)?;
    Ok(ws.on_upgrade(move |socket| session(socket, app, model, voice, query, admission)))
}

async fn session(
    mut socket: WebSocket,
    app: App,
    model: String,
    voice: String,
    query: SpeechQuery,
    _admission: crate::server::health::Admission,
) {
    let (audio_tx, mut audio_rx) = mpsc::channel(4);
    let mut stream: Option<AsyncStream> = None;
    let mut initial: Option<Speech> = None;
    let mut pump = None;
    let mut finalized = false;
    let deadline = tokio::time::sleep(query.inactivity_timeout);
    tokio::pin!(deadline);
    loop {
        tokio::select! {
            frame = socket.next() => match frame {
                Some(Ok(Message::Text(raw))) => {
                    deadline.as_mut().reset(tokio::time::Instant::now() + query.inactivity_timeout);
                    let result = if let Some(active) = &stream {
                        dispatch(&raw, active, initial.as_ref().expect("stream has initial settings")).await
                    } else {
                        initialize(&raw, &app, &model, &voice, &query, &audio_tx).await.map(|(created, speech, task)| {
                            stream = Some(created);
                            initial = Some(speech);
                            pump = Some(task);
                            false
                        })
                    };
                    match result {
                        Ok(done) if done => {
                            finalized = stream.as_ref().expect("initialized stream").finish().await.is_ok();
                        }
                        Ok(_) => {}
                        Err(error) => { policy_close(&mut socket, &error).await; break }
                    }
                }
                Some(Ok(Message::Binary(_))) => { policy_close(&mut socket, "binary messages are not supported").await; break }
                Some(Ok(Message::Close(_))) | None | Some(Err(_)) => break,
                _ => {}
            },
            delivery = audio_rx.recv(), if stream.is_some() => match delivery {
                Some(Ok((Some(chunk), _))) => {
                    let alignment = if query.sync_alignment { chunk.alignment.clone() } else { None };
                    let Ok(bytes) = encode(chunk, query.output_format) else { break };
                    if send_value(&mut socket, serde_json::to_value(WsAudio::audio(bytes, alignment)).expect("serializable response")).await.is_err() { break }
                }
                Some(Ok((None, true))) => {
                    let _ = send_value(&mut socket, serde_json::to_value(WsAudio::final_message()).expect("serializable response")).await;
                    let _ = socket.send(Message::Close(Some(CloseFrame { code: 1000, reason: "complete".into() }))).await;
                    break;
                }
                Some(Err(_)) | None => break,
                Some(Ok((None, false))) => {}
            },
            () = &mut deadline, if stream.is_some() && !finalized => {
                policy_close(&mut socket, "inactivity timeout").await; break
            }
        }
    }
    if let Some(stream) = stream {
        if !finalized {
            let _ = stream.cancel().await;
        }
        let _ = stream.close().await;
    }
    if let Some(pump) = pump {
        pump.abort();
    }
}

async fn initialize(
    raw: &str,
    app: &App,
    model: &str,
    voice: &str,
    query: &SpeechQuery,
    output: &mpsc::Sender<crate::Result<(Option<crate::AudioChunk>, bool)>>,
) -> Result<(AsyncStream, Speech, tokio::task::JoinHandle<()>), String> {
    let value = serde_json::from_str::<Value>(raw).map_err(|_| "message must be valid JSON")?;
    let mut object = value.as_object().cloned().ok_or("message must be a JSON object")?;
    if object.get("text") != Some(&Value::String(" ".into())) {
        return Err("first message text must be exactly \" \"".into());
    }
    for field in ["xi-api-key", "authorization", "pronunciation_dictionary_locators"] {
        object.remove(field);
    }
    let speech = parse_speech(Value::Object(object)).map_err(validation_message)?;
    let alignment = if query.sync_alignment { AlignmentType::Char } else { AlignmentType::None };
    let stream =
        app.engine.create_stream(model, voice, query.stream_params(&speech, alignment)).await.map_err(|error| error.to_string())?;
    let audio_stream = stream.clone();
    let output = output.clone();
    let pump = tokio::spawn(async move {
        loop {
            let delivery = audio_stream.recv_marked().await;
            let finished = matches!(delivery, Ok((None, true)) | Err(_));
            if output.send(delivery).await.is_err() || finished {
                break;
            }
        }
    });
    Ok((stream, speech, pump))
}

async fn dispatch(raw: &str, stream: &AsyncStream, initial: &Speech) -> Result<bool, String> {
    let value = serde_json::from_str::<Value>(raw).map_err(|_| "message must be valid JSON")?;
    let object = value.as_object().ok_or("message must be a JSON object")?;
    let allowed = ["text", "try_trigger_generation", "voice_settings", "generator_config", "flush"];
    if let Some(field) = object.keys().filter(|key| !allowed.contains(&key.as_str())).min() {
        return Err(format!("unsupported message field: {field}"));
    }
    let text = object.get("text").and_then(Value::as_str).ok_or("text is required and must be a string")?;
    validate_settings(object, text, initial)?;
    let flush = optional_bool(object, "flush")?;
    let trigger = optional_bool(object, "try_trigger_generation")?;
    if text.is_empty() {
        return Ok(true);
    }
    if !text.ends_with(' ') {
        return Err("text must end in a space".into());
    }
    if text != " " {
        stream.add_text(text).await.map_err(|error| error.to_string())?;
    }
    if flush {
        stream.force_generate().await.map_err(|error| error.to_string())?;
    } else if trigger {
        stream.try_generate().await.map_err(|error| error.to_string())?;
    }
    Ok(false)
}

fn validate_settings(object: &Map<String, Value>, text: &str, initial: &Speech) -> Result<(), String> {
    if let Some(settings) = object.get("voice_settings") {
        let parsed = parse_speech(json!({"text":text,"voice_settings":settings})).map_err(validation_message)?;
        if parsed.voice_settings != initial.voice_settings {
            return Err("voice_settings cannot change after initialization".into());
        }
    }
    if let Some(config) = object.get("generator_config") {
        let parsed = parse_speech(json!({"text":text,"generation_config":config})).map_err(validation_message)?;
        if parsed.chunk_length_schedule != initial.chunk_length_schedule {
            return Err("generator_config cannot change after initialization".into());
        }
    }
    Ok(())
}

fn optional_bool(object: &Map<String, Value>, field: &str) -> Result<bool, String> {
    match object.get(field) {
        None => Ok(false),
        Some(Value::Bool(value)) => Ok(*value),
        Some(_) => Err(format!("{field} must be boolean")),
    }
}

fn validation_message(error: WebError) -> String {
    match error {
        WebError::Validation(message) => message,
        _ => "invalid speech request".into(),
    }
}

async fn send_value(socket: &mut WebSocket, value: Value) -> Result<(), axum::Error> {
    socket.send(Message::Text(value.to_string().into())).await
}

async fn policy_close(socket: &mut WebSocket, error: &str) {
    let _ = send_value(socket, json!({"error":error})).await;
    let _ = socket.send(Message::Close(Some(CloseFrame { code: 1008, reason: error.into() }))).await;
}
