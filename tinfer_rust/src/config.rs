use std::net::SocketAddr;
use std::path::PathBuf;
use std::str::FromStr;

use serde::de::Error as _;
use serde::{Deserialize, Deserializer};

use crate::StreamParams;
use crate::audio::AudioFormat;
use crate::{AlignmentType, Error, Result};

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum Backend {
    Onnx,
    Tensorrt,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Device {
    Auto,
    Cpu,
    Cuda(u32),
}

impl<'de> Deserialize<'de> for Device {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> std::result::Result<Self, D::Error> {
        let value = String::deserialize(deserializer)?;
        match value.as_str() {
            "auto" => Ok(Self::Auto),
            "cpu" => Ok(Self::Cpu),
            _ => value
                .strip_prefix("cuda:")
                .ok_or_else(|| D::Error::custom("device must be auto, cpu, or cuda:N"))?
                .parse()
                .map(Self::Cuda)
                .map_err(|_| D::Error::custom("CUDA device index must be an integer")),
        }
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct ModelConfig {
    pub id: String,
    pub model: String,
    pub path: PathBuf,
    pub backend: Backend,
    pub device: Device,
    pub max_batch: usize,
    pub settings: serde_json::Value,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EngineSettings {
    pub engine_timeout_ms: u64,
    pub queue_capacity: usize,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SynthesisSettings {
    pub chunk_length_schedule: Vec<usize>,
    pub inactivity_timeout_ms: u64,
    pub alignment_type: AlignmentType,
}

impl SynthesisSettings {
    pub fn stream_params(&self) -> StreamParams {
        StreamParams {
            chunk_length_schedule: self.chunk_length_schedule.clone(),
            timeout: std::time::Duration::from_millis(self.inactivity_timeout_ms),
            alignment_type: self.alignment_type,
            model: serde_json::Value::Object(Default::default()),
        }
    }
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GrpcSettings {
    pub enabled: bool,
    pub address: SocketAddr,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct WebSettings {
    pub enabled: bool,
    pub websocket_enabled: bool,
    pub address: SocketAddr,
    pub output_format: String,
    pub websocket_inactivity_timeout_seconds: u64,
    pub sync_alignment: bool,
    pub auto_mode: bool,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Config {
    pub engine: EngineSettings,
    pub defaults: SynthesisSettings,
    pub grpc: GrpcSettings,
    pub web: WebSettings,
    pub models: Vec<ModelConfig>,
}

impl Config {
    pub fn validate(&self) -> Result<()> {
        let invalid_schedule = self.defaults.chunk_length_schedule.is_empty()
            || self.defaults.chunk_length_schedule.contains(&0)
            || self.defaults.chunk_length_schedule.windows(2).any(|pair| pair[0] > pair[1]);
        let invalid_model = self.models.iter().any(|model| {
            model.id.is_empty()
                || model.model.is_empty()
                || model.path.as_os_str().is_empty()
                || model.max_batch == 0
                || !model.settings.is_object()
        });
        let invalid_web = self.web.websocket_enabled && !self.web.enabled
            || !(1..=180).contains(&self.web.websocket_inactivity_timeout_seconds)
            || AudioFormat::from_str(&self.web.output_format).is_err();
        if self.engine.queue_capacity == 0 || invalid_schedule || invalid_model || invalid_web || !(self.grpc.enabled || self.web.enabled) {
            return Err(Error::Validation("invalid server configuration".into()));
        }
        Ok(())
    }
}
