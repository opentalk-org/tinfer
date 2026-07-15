mod server;
mod wire;

pub use server::{GrpcConfig, GrpcServer};

use std::pin::Pin;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use futures_util::Stream;
use tokio_stream::StreamExt;
use tokio_stream::wrappers::ReceiverStream;
use tonic::{Request, Response, Status};

use self::wire::{Content, content, options, response, status, synthesis};
use super::health::HealthState;
use crate::{AsyncEngine, AsyncStream, Error};

#[allow(clippy::doc_lazy_continuation)]
pub mod pb {
    tonic::include_proto!("styletts.v1");
}

pub(crate) struct Service {
    pub engine: AsyncEngine,
    pub health: Arc<HealthState>,
}

type Responses = Pin<Box<dyn Stream<Item = Result<pb::SynthesizeResponse, Status>> + Send>>;

#[tonic::async_trait]
impl pb::style_tts_service_server::StyleTtsService for Service {
    type SynthesizeStreamStream = Responses;
    type SynthesizeIncrementalStream = Responses;

    async fn health(&self, _: Request<pb::HealthRequest>) -> Result<Response<pb::HealthResponse>, Status> {
        let ready = self.health.ready();
        Ok(Response::new(pb::HealthResponse { ready, status: if ready { 1 } else { 2 } }))
    }

    async fn list_models(&self, _: Request<pb::ListModelsRequest>) -> Result<Response<pb::ListModelsResponse>, Status> {
        let models = self
            .engine
            .get_model_infos()
            .await
            .map_err(status)?
            .into_iter()
            .map(|info| pb::ModelInfo {
                model_id: info.model_id,
                supported_languages: info.supported_languages,
                default_language: info.default_language,
            })
            .collect();
        Ok(Response::new(pb::ListModelsResponse { models }))
    }

    async fn list_voices(&self, request: Request<pb::ListVoicesRequest>) -> Result<Response<pb::ListVoicesResponse>, Status> {
        let model = request.into_inner().model_id;
        let models = if model.is_empty() { self.engine.get_model_ids().await.map_err(status)? } else { vec![model] };
        let mut voices = Vec::new();
        for model_id in models {
            voices.extend(
                self.engine
                    .get_voice_ids(&model_id)
                    .await
                    .map_err(status)?
                    .into_iter()
                    .map(|voice_id| pb::VoiceInfo { model_id: model_id.clone(), voice_id }),
            );
        }
        Ok(Response::new(pb::ListVoicesResponse { voices }))
    }

    async fn synthesize(&self, request: Request<pb::SynthesizeRequest>) -> Result<Response<pb::SynthesizeResponse>, Status> {
        let _admission = self.health.admit().ok_or_else(|| Status::unavailable("server is not accepting synthesis requests"))?;
        let (text, model, voice, rate, params) = synthesis(request.into_inner(), self.engine.stream_params())?;
        let chunk = self.engine.generate_full(&model, &voice, &text, params).await.map_err(status)?;
        Ok(Response::new(response(chunk, rate)?))
    }

    async fn synthesize_stream(&self, request: Request<pb::SynthesizeRequest>) -> Result<Response<Self::SynthesizeStreamStream>, Status> {
        let admission = self.health.admit().ok_or_else(|| Status::unavailable("server is not accepting synthesis requests"))?;
        let (text, model, voice, rate, params) = synthesis(request.into_inner(), self.engine.stream_params())?;
        let stream = self.engine.start_stream(&model, &voice, &text, params).await.map_err(status)?;
        Ok(Response::new(output(stream, rate, admission)))
    }

    async fn synthesize_incremental(
        &self,
        request: Request<tonic::Streaming<pb::IncrementalSynthesizeRequest>>,
    ) -> Result<Response<Self::SynthesizeIncrementalStream>, Status> {
        let admission = self.health.admit().ok_or_else(|| Status::unavailable("server is not accepting synthesis requests"))?;
        let mut input = request.into_inner();
        let first = input.next().await.ok_or_else(|| Status::failed_precondition("config must be first"))??;
        let config = match first.content {
            Some(pb::incremental_synthesize_request::Content::Config(config)) => config,
            _ => return Err(Status::failed_precondition("config must be first")),
        };
        let (model, voice, rate, params) = options(config, self.engine.stream_params())?;
        let stream = self.engine.create_stream(&model, &voice, params).await.map_err(status)?;
        let (tx, rx) = tokio::sync::mpsc::channel(2);
        let done = Arc::new(AtomicBool::new(false));
        let outgoing = stream.clone();
        let output_done = done.clone();
        let output_tx = tx.clone();
        tokio::spawn(async move {
            let _admission = admission;
            loop {
                match outgoing.recv().await {
                    Ok(Some(chunk)) => {
                        if output_tx.send(response(chunk, rate)).await.is_err() {
                            let _ = outgoing.cancel().await;
                            break;
                        }
                    }
                    Ok(None) if output_done.load(Ordering::Acquire) => break,
                    Ok(None) => {}
                    Err(Error::Cancelled) if output_done.load(Ordering::Acquire) => break,
                    Err(error) => {
                        let _ = output_tx.send(Err(status(error))).await;
                        break;
                    }
                }
            }
            let _ = outgoing.close().await;
        });
        tokio::spawn(async move {
            while let Some(message) = input.next().await {
                let result = match message.and_then(content) {
                    Ok(Content::Text(text)) => stream.add_text(&text).await,
                    Ok(Content::Force) => stream.force_generate().await,
                    Ok(Content::Cancel) => stream.cancel().await,
                    Ok(Content::Config) => Err(Error::Validation("config can only be sent once".into())),
                    Err(error) => Err(Error::Validation(error.message().into())),
                };
                if let Err(error) = result {
                    done.store(true, Ordering::Release);
                    let _ = tx.send(Err(status(error))).await;
                    let _ = stream.cancel().await;
                    return;
                }
            }
            done.store(true, Ordering::Release);
            let _ = stream.finish().await;
        });
        Ok(Response::new(Box::pin(ReceiverStream::new(rx))))
    }
}

fn output(stream: AsyncStream, rate: u32, admission: super::health::Admission) -> Responses {
    let (tx, rx) = tokio::sync::mpsc::channel(2);
    tokio::spawn(async move {
        let _admission = admission;
        loop {
            match stream.recv().await {
                Ok(Some(chunk)) => {
                    if tx.send(response(chunk, rate)).await.is_err() {
                        let _ = stream.cancel().await;
                        break;
                    }
                }
                Ok(None) => break,
                Err(error) => {
                    let _ = tx.send(Err(status(error))).await;
                    break;
                }
            }
        }
        let _ = stream.close().await;
    });
    Box::pin(ReceiverStream::new(rx))
}
