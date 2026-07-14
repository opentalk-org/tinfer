use std::sync::Arc;

use crate::{Error, ModelConfig, Result};

pub mod base;
mod stub;
mod styletts2;

pub use base::Model;

pub fn load(config: &ModelConfig) -> Result<Arc<dyn Model>> {
    match config.model.as_str() {
        "stub" => stub::load(config),
        "styletts2" => styletts2::load(config),
        model => Err(Error::Catalog(format!("unknown model implementation: {model}"))),
    }
}
