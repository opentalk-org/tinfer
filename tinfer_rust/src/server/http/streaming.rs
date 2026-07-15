use std::collections::HashMap;
use std::convert::Infallible;

use axum::Json;
use axum::body::{Body, Bytes};
use axum::extract::{Path, Query, State};
use axum::http::header;
use axum::response::{IntoResponse, Response};
use base64::Engine as _;
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;

use super::{Transport, parse_query, parse_speech};
use crate::AlignmentType;
use crate::AudioChunk;
use crate::audio::{AudioEncoder, AudioEncoding};
use crate::server::web::wire::{TimedAudio, Timing, WebError};
use crate::server::web::{App, resolve_language, resolve_model};

pub(crate) async fn speech_stream(
    State(app): State<App>,
    Path(voice): Path<String>,
    Query(query): Query<HashMap<String, String>>,
    Json(value): Json<serde_json::Value>,
) -> Result<Response, WebError> {
    stream(app, voice, query, value, false).await
}

pub(crate) async fn timing_stream(
    State(app): State<App>,
    Path(voice): Path<String>,
    Query(query): Query<HashMap<String, String>>,
    Json(value): Json<serde_json::Value>,
) -> Result<Response, WebError> {
    stream(app, voice, query, value, true).await
}

async fn stream(
    app: App,
    voice: String,
    query: HashMap<String, String>,
    value: serde_json::Value,
    timed: bool,
) -> Result<Response, WebError> {
    let mut request = parse_speech(value)?;
    if request.text.trim().is_empty() {
        return Err(WebError::Validation("text must contain speech content".into()));
    }
    let parsed = parse_query(&query, Transport::Http)?;
    if parsed.output_format.encoding == AudioEncoding::WavPcm16 {
        return Err(WebError::Validation("WAV output is not supported for streaming".into()));
    }
    let admission = app.health.admit().ok_or(WebError::Unavailable)?;
    let model = resolve_model(&app.engine, request.model_id.clone()).await?;
    request.language_code = Some(resolve_language(&app.engine, &model, request.language_code.as_deref()).await?);
    let params = request.stream_params(if timed { AlignmentType::Char } else { AlignmentType::None });
    let stream = app.engine.create_stream(&model, &voice, params).await?;
    stream.add_text(&request.text).await?;
    stream.finish().await?;
    let (sender, receiver) = mpsc::channel::<Result<Bytes, Infallible>>(4);
    tokio::spawn(async move {
        let _admission = admission;
        let mut encoder: Option<AudioEncoder> = None;
        while let Ok((Some(chunk), _)) = stream.recv_marked().await {
            let current =
                encoder.get_or_insert_with(|| AudioEncoder::new(parsed.output_format, chunk.sample_rate).expect("validated encoder"));
            let payload = match encode_record(current, chunk, timed) {
                Ok(payload) => payload,
                Err(_) => break,
            };
            if sender.send(Ok(payload)).await.is_err() {
                let _ = stream.cancel().await;
                break;
            }
        }
        if let (false, Some(encoder)) = (timed, encoder.as_mut()) {
            match encoder.finish() {
                Ok(tail) if !tail.is_empty() => {
                    let _ = sender.send(Ok(tail)).await;
                }
                Ok(_) | Err(_) => {}
            }
        }
        let _ = stream.close().await;
    });
    let content_type = if timed { "text/event-stream" } else { mime(parsed.output_format.encoding) };
    Ok(([(header::CONTENT_TYPE, content_type)], Body::from_stream(ReceiverStream::new(receiver))).into_response())
}

fn encode_record(encoder: &mut AudioEncoder, chunk: AudioChunk, timed: bool) -> Result<Bytes, WebError> {
    let bytes = encoder.push(&chunk.audio).map_err(WebError::Audio)?;
    if !timed {
        return Ok(bytes);
    }
    let timing = Timing::from(chunk.alignment.unwrap_or_default());
    let record = TimedAudio {
        audio_base64: base64::engine::general_purpose::STANDARD.encode(bytes),
        alignment: timing.clone(),
        normalized_alignment: timing,
    };
    let mut json = serde_json::to_vec(&record).expect("timed response is serializable");
    json.push(b'\n');
    Ok(Bytes::from(json))
}

fn mime(encoding: AudioEncoding) -> &'static str {
    match encoding {
        AudioEncoding::Mp3 => "audio/mpeg",
        AudioEncoding::Opus => "audio/ogg",
        AudioEncoding::WavPcm16 => "audio/wav",
        _ => "application/octet-stream",
    }
}
