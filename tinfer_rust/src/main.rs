use std::time::Duration;

use tinfer_rust::server::{GrpcConfig, GrpcServer, HealthState, WebConfig, WebServer};
use tinfer_rust::{AsyncEngine, Config, Engine};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = std::env::args_os().nth(1).ok_or("usage: tinfer_rust CONFIG.yaml")?;
    let config: Config = serde_yaml::from_str(&std::fs::read_to_string(path)?)?;
    let engine = Engine::new(config.clone())?;
    let async_engine = AsyncEngine::new(engine.clone());
    let health = std::sync::Arc::new(HealthState::new());
    let grpc = GrpcServer::new(
        async_engine.clone(),
        health.clone(),
        GrpcConfig { address: config.grpc_address, shutdown_grace: Duration::from_secs(5) },
    );
    let web = WebServer::new(async_engine, health, WebConfig { address: config.http_address, shutdown_grace: Duration::from_secs(5) });
    grpc.start().await?;
    web.start().await?;
    tokio::signal::ctrl_c().await?;
    web.stop().await?;
    grpc.stop().await?;
    engine.stop()?;
    Ok(())
}
