use std::time::Duration;

use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use serde::Deserialize;
use tinfer_rust::{AlignmentType, AudioChunk, Engine, ModelConfig, Stream, StreamParams};

#[pyclass]
struct NativeEngine(Engine);

#[pyclass]
struct NativeStream(Stream);

#[pyclass(get_all)]
struct NativeChunk {
    audio: Vec<f32>,
    sample_rate: u32,
    chunk_index: u64,
    text_span: (usize, usize),
    alignment: Vec<(String, usize, usize, u64, u64)>,
}

#[derive(Default, Deserialize)]
struct Params {
    chunk_length_schedule: Option<Vec<usize>>,
    inactivity_timeout_ms: Option<f64>,
    alignment_type: Option<String>,
    tts_params: Option<serde_json::Value>,
}

#[pymethods]
impl NativeEngine {
    #[new]
    fn new(yaml: &str) -> PyResult<Self> {
        let config = serde_yaml::from_str(yaml).map_err(value_error)?;
        Engine::new(config).map(Self).map_err(runtime_error)
    }

    fn load_model(&self, yaml: &str) -> PyResult<()> {
        let config: ModelConfig = serde_yaml::from_str(yaml).map_err(value_error)?;
        self.0.load_model(config).map_err(runtime_error)
    }

    fn unload_model(&self, model_id: &str) -> PyResult<()> {
        self.0.unload_model(model_id).map_err(runtime_error)
    }

    fn get_model_ids(&self) -> PyResult<Vec<String>> {
        self.0.get_model_ids().map_err(runtime_error)
    }

    fn get_model_infos(&self) -> PyResult<Vec<(String, Vec<String>, String)>> {
        self.0
            .get_model_infos()
            .map(|models| models.into_iter().map(|info| (info.model_id, info.supported_languages, info.default_language)).collect())
            .map_err(runtime_error)
    }

    fn get_voice_ids(&self, model_id: &str) -> PyResult<Vec<String>> {
        self.0.get_voice_ids(model_id).map_err(runtime_error)
    }

    fn create_stream(&self, model: &str, voice: &str, params: &str) -> PyResult<NativeStream> {
        self.0.create_stream(model, voice, parse_params(&self.0, params)?).map(NativeStream).map_err(runtime_error)
    }

    fn generate_full(&self, model: &str, voice: &str, text: &str, params: &str) -> PyResult<NativeChunk> {
        self.0.generate_full(model, voice, text, parse_params(&self.0, params)?).map(NativeChunk::from).map_err(runtime_error)
    }

    fn stop(&self) -> PyResult<()> {
        self.0.stop().map_err(runtime_error)
    }
}

#[pymethods]
impl NativeStream {
    fn add_text(&self, text: &str) -> PyResult<()> {
        self.0.add_text(text).map_err(runtime_error)
    }
    fn force_generate(&self) -> PyResult<()> {
        self.0.force_generate().map_err(runtime_error)
    }
    fn try_generate(&self) -> PyResult<()> {
        self.0.try_generate().map_err(runtime_error)
    }
    fn cancel(&self) -> PyResult<()> {
        self.0.cancel().map_err(runtime_error)
    }
    fn recv(&self) -> PyResult<Option<NativeChunk>> {
        self.0.recv().map(|chunk| chunk.map(NativeChunk::from)).map_err(runtime_error)
    }
    fn get_audio(&self) -> PyResult<Vec<NativeChunk>> {
        self.0.get_audio().map(|chunks| chunks.into_iter().map(NativeChunk::from).collect()).map_err(runtime_error)
    }
    fn get_state(&self) -> PyResult<String> {
        self.0.get_state().map(|state| state.to_string()).map_err(runtime_error)
    }
    fn close(&self) -> PyResult<()> {
        self.0.close().map_err(runtime_error)
    }
}

impl From<AudioChunk> for NativeChunk {
    fn from(chunk: AudioChunk) -> Self {
        let alignment = chunk
            .alignment
            .into_iter()
            .flat_map(|alignment| alignment.items)
            .map(|item| (item.item, item.char_start, item.char_end, item.start_ms, item.end_ms))
            .collect();
        Self {
            audio: chunk.audio,
            sample_rate: chunk.sample_rate,
            chunk_index: chunk.chunk_index,
            text_span: (chunk.text_span.start, chunk.text_span.end),
            alignment,
        }
    }
}

fn parse_params(engine: &Engine, json: &str) -> PyResult<StreamParams> {
    let value: Params = serde_json::from_str(json).map_err(value_error)?;
    let mut params = engine.stream_params();
    if let Some(schedule) = value.chunk_length_schedule {
        params.chunk_length_schedule = schedule;
    }
    if let Some(timeout) = value.inactivity_timeout_ms {
        params.timeout = Duration::from_secs_f64(timeout / 1000.0);
    }
    if let Some(alignment) = value.alignment_type {
        params.alignment_type = match alignment.as_str() {
            "word" => AlignmentType::Word,
            "char" => AlignmentType::Char,
            "phoneme" => AlignmentType::Phoneme,
            "none" => AlignmentType::None,
            _ => return Err(PyValueError::new_err("invalid alignment_type")),
        };
    }
    params.model = value.tts_params.unwrap_or_default();
    Ok(params)
}

fn value_error(error: impl ToString) -> PyErr {
    PyValueError::new_err(error.to_string())
}

fn runtime_error(error: impl ToString) -> PyErr {
    PyRuntimeError::new_err(error.to_string())
}

#[pymodule]
fn _native(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<NativeEngine>()?;
    module.add_class::<NativeStream>()?;
    module.add_class::<NativeChunk>()
}
