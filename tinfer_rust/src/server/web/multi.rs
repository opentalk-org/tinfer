use std::collections::HashMap;

use axum::extract::ws::{CloseFrame, Message, WebSocket, WebSocketUpgrade};
use axum::extract::{Path, Query, State};
use axum::response::Response;
use futures_util::StreamExt;
use serde_json::{Map, Value};
use tokio::sync::mpsc;

use super::wire::{WebError, WsAudio, encode};
use super::{App, resolve_language, resolve_model};
use crate::server::http::{SpeechQuery, Transport, parse_query, parse_speech};
use crate::{AlignmentType, AsyncStream, Error};

type Delivery = crate::Result<(Option<crate::AudioChunk>, bool)>;
type Output = mpsc::Sender<(String, Delivery)>;

pub(super) async fn upgrade(
    ws: WebSocketUpgrade,
    State(app): State<App>,
    Path(voice): Path<String>,
    Query(query): Query<HashMap<String, String>>,
) -> Result<Response, WebError> {
    let mut parsed = parse_query(&query, Transport::WebSocket)?;
    let model = resolve_model(&app.engine, query.get("model_id").cloned()).await?;
    if !app.engine.get_voice_ids(&model).await?.contains(&voice) {
        return Err(Error::Catalog(format!("unknown voice: {voice}")).into());
    }
    parsed.language_code = Some(resolve_language(&app.engine, &model, parsed.language_code.as_deref()).await?);
    let admission = app.health.admit().ok_or(WebError::Unavailable)?;
    Ok(ws.on_upgrade(move |socket| session(socket, app, model, voice, parsed, admission)))
}

async fn session(
    mut socket: WebSocket,
    app: App,
    model: String,
    voice: String,
    query: crate::server::http::SpeechQuery,
    _admission: crate::server::health::Admission,
) {
    let (output_tx, mut output_rx) = mpsc::channel(8);
    let mut contexts = HashMap::<String, AsyncStream>::new();
    let mut initialized = false;
    let mut closing = false;
    loop {
        tokio::select! {
            frame = socket.next(), if !closing => match frame {
                Some(Ok(Message::Text(raw))) => {
                    let Ok(value) = serde_json::from_str::<Value>(&raw) else { policy_close(&mut socket, "message must be valid JSON").await; break };
                    let Some(command) = value.as_object() else { policy_close(&mut socket, "message must be a JSON object").await; break };
                    if let Err(error) = dispatch(command, &app, &model, &voice, &query, &output_tx, &mut contexts, &mut initialized, &mut closing).await {
                        policy_close(&mut socket, &error).await; break
                    }
                }
                Some(Ok(Message::Binary(_))) => { policy_close(&mut socket, "binary messages are not supported").await; break }
                Some(Ok(Message::Close(_))) | Some(Err(_)) | None => break,
                _ => {}
            },
            output = output_rx.recv() => match output {
                Some((context_id, Ok((Some(chunk), _)))) => {
                    let alignment = if query.sync_alignment { chunk.alignment.clone() } else { None };
                    let Ok(bytes) = encode(chunk, query.output_format) else { break };
                    let mut value = serde_json::to_value(WsAudio::audio(bytes, alignment)).expect("serializable response");
                    value.as_object_mut().expect("response object").insert("contextId".into(), Value::String(context_id));
                    if send_value(&mut socket, value).await.is_err() { break }
                }
                Some((context_id, Ok((None, true)))) => {
                    if let Some(stream) = contexts.remove(&context_id) { let _ = stream.close().await; }
                    if send_value(&mut socket, serde_json::json!({"contextId":context_id,"isFinal":true})).await.is_err() { break }
                    if closing && contexts.is_empty() {
                        let _ = socket.send(Message::Close(Some(CloseFrame { code: 1000, reason: "complete".into() }))).await;
                        break
                    }
                }
                Some((context_id, Err(_))) => {
                    contexts.remove(&context_id);
                    let _ = send_value(&mut socket, serde_json::json!({"contextId":context_id,"error":"audio pump failed"})).await;
                }
                Some((_context_id, Ok((None, false)))) => {}
                None => break,
            }
        }
    }
    for stream in contexts.into_values() {
        let _ = stream.cancel().await;
        let _ = stream.close().await;
    }
}

#[allow(clippy::too_many_arguments)]
async fn dispatch(
    command: &Map<String, Value>,
    app: &App,
    model: &str,
    voice: &str,
    query: &crate::server::http::SpeechQuery,
    output: &Output,
    contexts: &mut HashMap<String, AsyncStream>,
    initialized: &mut bool,
    closing: &mut bool,
) -> Result<(), String> {
    let allowed = [
        "text",
        "context_id",
        "voice_settings",
        "generation_config",
        "pronunciation_dictionary_locators",
        "xi_api_key",
        "authorization",
        "flush",
        "close_context",
        "close_socket",
    ];
    if let Some(field) = command.keys().filter(|key| !allowed.contains(&key.as_str())).min() {
        return Err(format!("unsupported message field: {field}"));
    }
    let text = command.get("text").and_then(Value::as_str);
    if !*initialized && text != Some(" ") {
        return Err("first message text must be exactly \" \"".into());
    }
    if optional_bool(command, "close_socket")? {
        *closing = true;
        for stream in contexts.values() {
            stream.finish().await.map_err(|error| error.to_string())?;
        }
        return Ok(());
    }
    let id = command.get("context_id").map_or(Ok("default"), |value| value.as_str().ok_or("context_id must be a string"))?.to_owned();
    if id.is_empty() {
        return Err("context_id must not be empty".into());
    }
    let close_context = optional_bool(command, "close_context")?;
    if close_context {
        let stream = contexts.get(&id).ok_or_else(|| format!("unknown context: {id}"))?;
        stream.finish().await.map_err(|error| error.to_string())?;
        return Ok(());
    }
    if contexts.contains_key(&id) && (command.contains_key("voice_settings") || command.contains_key("generation_config")) {
        let stream = contexts.remove(&id).expect("known context");
        stream.finish().await.map_err(|error| error.to_string())?;
        create_context(command, &id, app, model, voice, query, output, contexts).await?;
        return Ok(());
    }
    if !contexts.contains_key(&id) {
        if text.is_none() {
            return Err(format!("unknown context: {id}"));
        }
        create_context(command, &id, app, model, voice, query, output, contexts).await?;
        *initialized = true;
        return Ok(());
    }
    let stream = contexts.get(&id).expect("context was created");
    if let Some(text) = text.filter(|text| !matches!(*text, "" | " ")) {
        if !text.ends_with(' ') {
            return Err("text must end in a space".into());
        }
        stream.add_text(text).await.map_err(|error| error.to_string())?;
    }
    if optional_bool(command, "flush")? {
        stream.force_generate().await.map_err(|error| error.to_string())?;
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
async fn create_context(
    command: &Map<String, Value>,
    id: &str,
    app: &App,
    model: &str,
    voice: &str,
    query: &SpeechQuery,
    output: &Output,
    contexts: &mut HashMap<String, AsyncStream>,
) -> Result<(), String> {
    let mut speech = Map::new();
    speech.insert("text".into(), command.get("text").cloned().unwrap_or_else(|| Value::String(" ".into())));
    for field in ["voice_settings", "generation_config"] {
        if let Some(value) = command.get(field) {
            speech.insert(field.into(), value.clone());
        }
    }
    let speech = parse_speech(Value::Object(speech)).map_err(validation_message)?;
    if !matches!(speech.text.as_str(), "" | " ") && !speech.text.ends_with(' ') {
        return Err("text must end in a space".into());
    }
    let alignment = if query.sync_alignment { AlignmentType::Char } else { AlignmentType::None };
    let stream =
        app.engine.create_stream(model, voice, query.stream_params(&speech, alignment)).await.map_err(|error| error.to_string())?;
    pump(id.to_owned(), stream.clone(), output.clone());
    if !matches!(speech.text.as_str(), "" | " ") {
        stream.add_text(&speech.text).await.map_err(|error| error.to_string())?;
        if optional_bool(command, "flush")? {
            stream.force_generate().await.map_err(|error| error.to_string())?;
        }
    }
    contexts.insert(id.to_owned(), stream);
    Ok(())
}

fn optional_bool(command: &Map<String, Value>, field: &str) -> Result<bool, String> {
    match command.get(field) {
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

fn pump(id: String, stream: AsyncStream, output: Output) {
    tokio::spawn(async move {
        loop {
            let delivery = stream.recv_marked().await;
            let finished = matches!(delivery, Ok((None, true)) | Err(_));
            if output.send((id.clone(), delivery)).await.is_err() || finished {
                break;
            }
        }
    });
}

async fn send_value(socket: &mut WebSocket, value: Value) -> Result<(), axum::Error> {
    socket.send(Message::Text(value.to_string().into())).await
}

async fn policy_close(socket: &mut WebSocket, error: &str) {
    let _ = send_value(socket, serde_json::json!({"error":error})).await;
    let _ = socket.send(Message::Close(Some(CloseFrame { code: 1008, reason: error.into() }))).await;
}
