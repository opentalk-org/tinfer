use std::net::{IpAddr, Ipv4Addr, SocketAddr};
use std::path::Path;
use std::sync::Arc;
use std::time::Duration;

use reqwest::Client;
use tinfer_rust::server::{HealthState, WebConfig, WebServer};
use tinfer_rust::{AsyncEngine, Backend, Config, Device, Engine, ModelConfig};

pub struct TestServer {
    pub address: SocketAddr,
    pub client: Client,
    server: WebServer,
    engine: Engine,
}

impl TestServer {
    pub async fn start() -> Self {
        let engine = Engine::new(Config {
            models: vec![ModelConfig {
                id: "stub".into(),
                model: "stub".into(),
                path: Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/artifacts/stub"),
                backend: Backend::Onnx,
                device: Device::Cpu,
                max_batch: 4,
            }],
            ..Config::default()
        })
        .unwrap();
        let server = WebServer::new(
            AsyncEngine::new(engine.clone()),
            Arc::new(HealthState::new()),
            WebConfig {
                address: SocketAddr::new(IpAddr::V4(Ipv4Addr::LOCALHOST), 0),
                shutdown_grace: Duration::from_secs(1),
            },
        );
        let address = server.start().await.unwrap();
        Self { address, client: Client::new(), server, engine }
    }

    pub fn url(&self, path: &str) -> String {
        format!("http://{}{}", self.address, path)
    }

    pub fn ws_url(&self, path: &str) -> String {
        format!("ws://{}{}", self.address, path)
    }

    pub async fn stop(self) {
        self.server.stop().await.unwrap();
        self.engine.stop().unwrap();
    }
}
