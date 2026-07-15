use std::ops::Range;
use std::time::Duration;

use serde::Deserialize;
use thiserror::Error;

#[derive(Clone, Debug, Eq, Error, PartialEq)]
pub enum Error {
    #[error("{0}")]
    Validation(String),
    #[error("{0}")]
    Catalog(String),
    #[error("request cancelled")]
    Cancelled,
    #[error("{0}")]
    Inference(String),
    #[error("engine stopped")]
    Shutdown,
}
pub type Result<T> = std::result::Result<T, Error>;

#[derive(Clone, Copy, Debug, Default, Deserialize, Eq, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum AlignmentType {
    #[default]
    Word,
    Char,
    Phoneme,
    None,
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct AlignmentItem {
    pub item: String,
    pub char_start: usize,
    pub char_end: usize,
    pub start_ms: u64,
    pub end_ms: u64,
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct Alignment {
    pub items: Vec<AlignmentItem>,
    pub kind: AlignmentType,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct AudioChunk {
    pub audio: Vec<f32>,
    pub sample_rate: u32,
    pub chunk_index: u64,
    pub text_span: Range<usize>,
    pub alignment: Option<Alignment>,
}

impl AudioChunk {
    pub fn merge(chunks: Vec<Self>) -> Result<Self> {
        let sample_rate = chunks.first().map_or(0, |chunk| chunk.sample_rate);
        if chunks.iter().any(|chunk| chunk.sample_rate != sample_rate) {
            return Err(Error::Inference("inconsistent sample rates".into()));
        }
        let end = chunks.iter().map(|chunk| chunk.text_span.end).max().unwrap_or(0);
        Ok(Self { audio: chunks.into_iter().flat_map(|chunk| chunk.audio).collect(), sample_rate, text_span: 0..end, ..Self::default() })
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ModelInfo {
    pub model_id: String,
    pub supported_languages: Vec<String>,
    pub default_language: String,
}

#[derive(Clone, Debug)]
pub struct StreamParams {
    pub chunk_length_schedule: Vec<usize>,
    pub timeout: Duration,
    pub alignment_type: AlignmentType,
    pub model: serde_json::Value,
}

#[derive(Clone, Debug)]
pub struct ModelRequest {
    pub stream_id: u64,
    pub operation: ModelOperation,
    pub text: String,
    pub voice_id: String,
    pub params: serde_json::Value,
    pub state: serde_json::Value,
    pub alignment_type: AlignmentType,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ModelOperation {
    Start,
    Continue,
}

#[derive(Clone, Debug)]
pub struct ModelOutput {
    pub audio: Vec<f32>,
    pub sample_rate: u32,
    pub alignment: Option<Alignment>,
    pub state: serde_json::Value,
    pub complete: bool,
}
