use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Duration;

use tokio::sync::{Mutex, watch};

use super::super::health::{HealthState, ServingState};
use crate::{AsyncEngine, Engine, Error, Result};

#[derive(Clone)]
pub struct WebConfig {
    pub settings: crate::WebSettings,
    pub shutdown_grace: Duration,
}

struct Running {
    shutdown: watch::Sender<bool>,
    done: watch::Receiver<bool>,
    abort: tokio::task::AbortHandle,
}

pub struct WebServer {
    engine: AsyncEngine,
    health: Arc<HealthState>,
    config: WebConfig,
    running: Mutex<Option<Running>>,
}

impl WebServer {
    pub fn new(engine: AsyncEngine, health: Arc<HealthState>, config: WebConfig) -> Self {
        Self { engine, health, config, running: Mutex::new(None) }
    }

    pub fn from_engine(engine: Engine, health: Arc<HealthState>, config: WebConfig) -> Self {
        Self::new(AsyncEngine::new(engine), health, config)
    }

    pub async fn start(&self) -> Result<SocketAddr> {
        let mut running = self.running.lock().await;
        assert!(running.is_none(), "web server is already running");
        let listener = tokio::net::TcpListener::bind(self.config.settings.address).await.map_err(io_error)?;
        let address = listener.local_addr().map_err(io_error)?;
        let (shutdown, mut stop) = watch::channel(false);
        let (finished, done) = watch::channel(false);
        let app = super::router(self.engine.clone(), self.health.clone(), self.config.settings.clone());
        let task = tokio::spawn(async move {
            let result = axum::serve(listener, app)
                .with_graceful_shutdown(async move { while !*stop.borrow() && stop.changed().await.is_ok() {} })
                .await;
            let _ = finished.send(true);
            result
        });
        *running = Some(Running { shutdown, done, abort: task.abort_handle() });
        self.health.set(ServingState::Ready);
        Ok(address)
    }

    pub async fn stop(&self) -> Result<()> {
        self.health.set(ServingState::Draining);
        let Some(running) = self.running.lock().await.take() else {
            self.health.set(ServingState::Stopped);
            return Ok(());
        };
        let _ = running.shutdown.send(true);
        let mut done = running.done;
        let completed = tokio::time::timeout(self.config.shutdown_grace, async {
            while !*done.borrow() && done.changed().await.is_ok() {}
            self.health.drained().await;
        })
        .await
        .is_ok();
        if !completed {
            running.abort.abort();
        }
        self.health.set(ServingState::Stopped);
        Ok(())
    }
}

fn io_error(error: std::io::Error) -> Error {
    Error::Inference(error.to_string())
}
