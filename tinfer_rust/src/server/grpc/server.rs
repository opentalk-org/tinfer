use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Duration;

use tokio::sync::{Mutex, watch};
use tokio_stream::wrappers::TcpListenerStream;
use tonic_health::ServingStatus;
use tonic_health::server::HealthReporter;

use super::super::health::{HealthState, ServingState};
use super::{Service, pb};
use crate::{AsyncEngine, Engine, Error, Result};

#[derive(Clone, Copy)]
pub struct GrpcConfig {
    pub address: SocketAddr,
    pub shutdown_grace: Duration,
}

struct Running {
    shutdown: watch::Sender<bool>,
    done: watch::Receiver<bool>,
    abort: tokio::task::AbortHandle,
    reporter: HealthReporter,
}

pub struct GrpcServer {
    engine: AsyncEngine,
    health: Arc<HealthState>,
    config: GrpcConfig,
    running: Mutex<Option<Running>>,
}

impl GrpcServer {
    pub fn new(engine: AsyncEngine, health: Arc<HealthState>, config: GrpcConfig) -> Self {
        Self { engine, health, config, running: Mutex::new(None) }
    }

    pub fn from_engine(engine: Engine, health: Arc<HealthState>, config: GrpcConfig) -> Self {
        Self::new(AsyncEngine::new(engine), health, config)
    }

    pub async fn start(&self) -> Result<SocketAddr> {
        let mut running = self.running.lock().await;
        assert!(running.is_none(), "gRPC server is already running");
        let listener = tokio::net::TcpListener::bind(self.config.address).await.map_err(|error| Error::Inference(error.to_string()))?;
        let address = listener.local_addr().map_err(|error| Error::Inference(error.to_string()))?;
        let (shutdown, mut stop) = watch::channel(false);
        let (finished, done) = watch::channel(false);
        let (reporter, health_service) = tonic_health::server::health_reporter();
        let service = Service { engine: self.engine.clone(), health: self.health.clone() };
        let task = tokio::spawn(async move {
            let result = tonic::transport::Server::builder()
                .add_service(health_service)
                .add_service(pb::style_tts_service_server::StyleTtsServiceServer::new(service))
                .serve_with_incoming_shutdown(TcpListenerStream::new(listener), async move {
                    let _ = stop.wait_for(|stop| *stop).await;
                })
                .await;
            let _ = finished.send(true);
            result
        });
        *running = Some(Running { shutdown, done, abort: task.abort_handle(), reporter });
        self.health.set(ServingState::Ready);
        Ok(address)
    }

    pub async fn stop(&self) -> Result<()> {
        let running = self.running.lock().await.take();
        let Some(mut running) = running else {
            self.health.set(ServingState::Stopped);
            return Ok(());
        };
        running.reporter.set_service_status("", ServingStatus::NotServing).await;
        self.health.set(ServingState::Draining);
        let deadline = tokio::time::Instant::now() + self.config.shutdown_grace;
        let drained = tokio::time::timeout_at(deadline, self.health.drained()).await.is_ok();
        let _ = running.shutdown.send(true);
        let mut done = running.done;
        let completed = drained && tokio::time::timeout_at(deadline, done.wait_for(|done| *done)).await.is_ok();
        if !completed {
            running.abort.abort();
        }
        self.health.set(ServingState::Stopped);
        Ok(())
    }
}
