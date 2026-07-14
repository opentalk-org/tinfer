use std::net::{IpAddr, Ipv4Addr, SocketAddr};
use std::path::PathBuf;

use serde::de::Error as _;
use serde::{Deserialize, Deserializer};

use crate::{Error, Result};

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
}

#[derive(Clone, Debug, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Config {
    pub models: Vec<ModelConfig>,
    pub chunk_length_schedule: Vec<usize>,
    pub timeout_ms: u64,
    pub queue_capacity: usize,
    pub grpc_address: SocketAddr,
    pub http_address: SocketAddr,
}

impl Default for Config {
    fn default() -> Self {
        let localhost = IpAddr::V4(Ipv4Addr::LOCALHOST);
        Self {
            models: Vec::new(),
            chunk_length_schedule: vec![80, 160, 250, 290],
            timeout_ms: 80,
            queue_capacity: 64,
            grpc_address: SocketAddr::new(localhost, 50_051),
            http_address: SocketAddr::new(localhost, 8_000),
        }
    }
}

impl Config {
    pub fn validate(&self) -> Result<()> {
        if self.queue_capacity == 0
            || self.chunk_length_schedule.is_empty()
            || self.chunk_length_schedule.contains(&0)
            || self.chunk_length_schedule.windows(2).any(|pair| pair[0] > pair[1])
            || self
                .models
                .iter()
                .any(|model| model.id.is_empty() || model.model.is_empty() || model.path.as_os_str().is_empty() || model.max_batch == 0)
        {
            return Err(Error::Validation("invalid engine configuration".into()));
        }
        Ok(())
    }
}
