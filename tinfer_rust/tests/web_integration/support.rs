use std::net::{IpAddr, Ipv4Addr, SocketAddr};
use std::sync::Arc;
use std::time::Duration;

use reqwest::Client;
use tinfer_rust::server::{HealthState, WebConfig, WebServer};
use tinfer_rust::{AsyncEngine, Engine};

use crate::common;

pub struct TestServer {
    pub address: SocketAddr,
    pub client: Client,
    pub health: Arc<HealthState>,
    pub server: Arc<WebServer>,
    engine: Engine,
}

impl TestServer {
    pub async fn start() -> Self {
        Self::start_with_websockets(true).await
    }

    pub async fn start_with_websockets(enabled: bool) -> Self {
        let engine = Engine::new(common::config(vec![common::stub_model()])).unwrap();
        let health = Arc::new(HealthState::new());
        let mut settings = common::web_settings(SocketAddr::new(IpAddr::V4(Ipv4Addr::LOCALHOST), 0));
        settings.websocket_enabled = enabled;
        let server = Arc::new(WebServer::new(
            AsyncEngine::new(engine.clone()),
            health.clone(),
            WebConfig { settings, shutdown_grace: Duration::from_secs(1) },
        ));
        let address = server.start().await.unwrap();
        Self { address, client: Client::new(), health, server, engine }
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
