use std::sync::Arc;

use crate::{AudioChunk, Engine, ModelConfig, ModelInfo, Result, Stream, StreamParams};

#[derive(Clone)]
pub struct AsyncEngine {
    engine: Engine,
}

#[derive(Clone)]
pub struct AsyncStream {
    stream: Arc<Stream>,
}

impl AsyncEngine {
    pub fn new(engine: Engine) -> Self {
        Self { engine }
    }

    pub async fn load_model(&self, config: ModelConfig) -> Result<()> {
        self.call(move |engine| engine.load_model(config)).await
    }

    pub async fn unload_model(&self, model: &str) -> Result<()> {
        let model = model.to_owned();
        self.call(move |engine| engine.unload_model(&model)).await
    }

    pub async fn get_model_ids(&self) -> Result<Vec<String>> {
        self.call(|engine| engine.get_model_ids()).await
    }

    pub async fn get_model_infos(&self) -> Result<Vec<ModelInfo>> {
        self.call(|engine| engine.get_model_infos()).await
    }

    pub async fn get_voice_ids(&self, model: &str) -> Result<Vec<String>> {
        let model = model.to_owned();
        self.call(move |engine| engine.get_voice_ids(&model)).await
    }

    pub fn stream_params(&self) -> StreamParams {
        self.engine.stream_params()
    }

    pub async fn create_stream(&self, model: &str, voice: &str, params: StreamParams) -> Result<AsyncStream> {
        let model = model.to_owned();
        let voice = voice.to_owned();
        self.call(move |engine| engine.create_stream(&model, &voice, params)).await.map(|stream| AsyncStream { stream: Arc::new(stream) })
    }

    pub async fn start_stream(&self, model: &str, voice: &str, text: &str, params: StreamParams) -> Result<AsyncStream> {
        let (model, voice, text) = (model.to_owned(), voice.to_owned(), text.to_owned());
        self.call(move |engine| engine.start_stream(&model, &voice, &text, params))
            .await
            .map(|stream| AsyncStream { stream: Arc::new(stream) })
    }

    pub async fn generate_full(&self, model: &str, voice: &str, text: &str, params: StreamParams) -> Result<AudioChunk> {
        let (model, voice, text) = (model.to_owned(), voice.to_owned(), text.to_owned());
        self.call(move |engine| engine.generate_full(&model, &voice, &text, params)).await
    }

    pub async fn stop(&self) -> Result<()> {
        self.call(|engine| engine.stop()).await
    }

    async fn call<T: Send + 'static>(&self, call: impl FnOnce(Engine) -> Result<T> + Send + 'static) -> Result<T> {
        let engine = self.engine.clone();
        blocking(move || call(engine)).await
    }
}

impl AsyncStream {
    pub async fn add_text(&self, text: &str) -> Result<()> {
        let text = text.to_owned();
        self.call(move |stream| stream.add_text(&text)).await
    }

    pub async fn force_generate(&self) -> Result<()> {
        self.call(|stream| stream.force_generate()).await
    }

    pub async fn try_generate(&self) -> Result<()> {
        self.call(|stream| stream.try_generate()).await
    }

    pub async fn finish(&self) -> Result<()> {
        self.call(|stream| stream.finish()).await
    }

    pub async fn cancel(&self) -> Result<()> {
        self.call(|stream| stream.cancel()).await
    }

    pub async fn recv(&self) -> Result<Option<AudioChunk>> {
        self.call(|stream| stream.recv()).await
    }

    pub(crate) async fn recv_marked(&self) -> Result<(Option<AudioChunk>, bool)> {
        self.call(|stream| stream.recv_marked()).await
    }

    pub async fn close(&self) -> Result<()> {
        self.call(|stream| stream.close()).await
    }

    async fn call<T: Send + 'static>(&self, call: impl FnOnce(Arc<Stream>) -> Result<T> + Send + 'static) -> Result<T> {
        let stream = self.stream.clone();
        blocking(move || call(stream)).await
    }
}

async fn blocking<T: Send + 'static>(call: impl FnOnce() -> Result<T> + Send + 'static) -> Result<T> {
    tokio::task::spawn_blocking(call).await.map_err(|error| crate::Error::Inference(error.to_string()))?
}
