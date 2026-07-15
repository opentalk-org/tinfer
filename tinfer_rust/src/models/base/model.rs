use crate::{ModelInfo, ModelOutput, ModelRequest, Result};

pub trait Model: Send + Sync {
    fn info(&self) -> &ModelInfo;
    fn voices(&self) -> &[String];
    fn generate_batch(&self, batch: &[ModelRequest]) -> Result<Vec<ModelOutput>>;
    fn close_stream(&self, stream_id: u64) -> Result<()>;
}
