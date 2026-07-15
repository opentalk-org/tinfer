#![allow(clippy::result_large_err)]

use tonic::Status;

use super::pb;
use crate::audio::{AudioEncoder, AudioEncoding, AudioFormat};
use crate::{AudioChunk, Error, StreamParams};

pub(crate) enum Content {
    Config,
    Text(String),
    Force,
    Cancel,
}

pub(crate) fn content(request: pb::IncrementalSynthesizeRequest) -> Result<Content, Status> {
    use pb::incremental_synthesize_request::Content as Pb;
    match request.content {
        Some(Pb::Config(_)) => Ok(Content::Config),
        Some(Pb::TextChunk(text)) => Ok(Content::Text(text)),
        Some(Pb::ForceSynthesis(_)) => Ok(Content::Force),
        Some(Pb::Cancel(_)) => Ok(Content::Cancel),
        _ => Err(Status::invalid_argument("incremental content is required")),
    }
}

pub(crate) fn synthesis(
    request: pb::SynthesizeRequest,
    defaults: StreamParams,
) -> Result<(String, String, String, u32, StreamParams), Status> {
    if request.text.trim().is_empty() {
        return Err(Status::invalid_argument("text must not be empty"));
    }
    let (model, voice, rate, params) = options(request.config.ok_or_else(|| Status::invalid_argument("config is required"))?, defaults)?;
    Ok((request.text, model, voice, rate, params))
}

pub(crate) fn options(config: pb::SynthesisConfig, mut params: StreamParams) -> Result<(String, String, u32, StreamParams), Status> {
    if config.model_id.is_empty() || config.voice_id.is_empty() {
        return Err(Status::invalid_argument("model_id and voice_id are required"));
    }
    if !matches!(config.sample_rate_hz, 8_000 | 16_000 | 22_050 | 24_000 | 44_100) {
        return Err(Status::invalid_argument("unsupported sample_rate_hz"));
    }
    if !config.language.is_empty() {
        params.model["language"] = serde_json::Value::String(config.language);
    }
    let rate = u32::try_from(config.sample_rate_hz).expect("supported sample rates are positive");
    Ok((config.model_id, config.voice_id, rate, params))
}

pub(crate) fn response(chunk: AudioChunk, rate: u32) -> Result<pb::SynthesizeResponse, Status> {
    let format = AudioFormat { encoding: AudioEncoding::Pcm16, sample_rate: rate, bitrate_kbps: None };
    let mut encoder = AudioEncoder::new(format, chunk.sample_rate).map_err(|error| Status::internal(error.to_string()))?;
    let mut audio_data = encoder.push(&chunk.audio).map_err(|error| Status::internal(error.to_string()))?.to_vec();
    audio_data.extend(encoder.finish().map_err(|error| Status::internal(error.to_string()))?);
    let alignments = chunk
        .alignment
        .into_iter()
        .flat_map(|alignment| alignment.items)
        .map(|item| pb::WordAlignment { word: item.item, start_ms: item.start_ms as i32, end_ms: item.end_ms as i32 })
        .collect();
    Ok(pb::SynthesizeResponse { audio_data, alignments })
}

pub(crate) fn status(error: Error) -> Status {
    match error {
        Error::Validation(message) => Status::invalid_argument(message),
        Error::Catalog(message) => Status::not_found(message),
        Error::Cancelled => Status::cancelled("request cancelled"),
        Error::Inference(_) => Status::internal("inference failed"),
        Error::Shutdown => Status::unavailable("engine stopped"),
    }
}
