use std::net::{IpAddr, Ipv4Addr, SocketAddr};
use std::path::Path;
use std::sync::Arc;
use std::time::Duration;

use base64::Engine as _;
use futures_util::{SinkExt, StreamExt};
use tinfer_rust::server::{HealthState, WebConfig, WebServer};
use tinfer_rust::{AsyncEngine, Backend, Config, Device, Engine, ModelConfig};
use tokio_tungstenite::tungstenite::Message;

#[tokio::test]
async fn elevenlabs_http_and_websocket_routes_run_over_loopback() {
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
        WebConfig { address: SocketAddr::new(IpAddr::V4(Ipv4Addr::LOCALHOST), 0), shutdown_grace: Duration::from_secs(1) },
    );
    let address = server.start().await.unwrap();
    let http = reqwest::Client::new();
    let base = format!("http://{address}");

    assert!(http.get(format!("{base}/health")).send().await.unwrap().status().is_success());
    let models: serde_json::Value = http.get(format!("{base}/v1/models")).send().await.unwrap().json().await.unwrap();
    assert_eq!(models[0]["model_id"], "stub");
    let voices: serde_json::Value = http.get(format!("{base}/v1/voices?model_id=stub")).send().await.unwrap().json().await.unwrap();
    assert_eq!(voices["voices"][0]["voice_id"], "default");

    let audio = http
        .post(format!("{base}/v1/text-to-speech/default?output_format=pcm_24000"))
        .json(&serde_json::json!({ "text": "hello", "model_id": "stub" }))
        .send()
        .await
        .unwrap();
    assert!(audio.status().is_success());
    assert!(!audio.bytes().await.unwrap().is_empty());

    let (mut websocket, _) = tokio_tungstenite::connect_async(format!(
        "ws://{address}/v1/text-to-speech/default/stream-input?model_id=stub&output_format=pcm_24000"
    ))
    .await
    .unwrap();
    websocket.send(Message::text(r#"{"text":" "}"#)).await.unwrap();
    websocket.send(Message::text(r#"{"text":"short ","try_trigger_generation":true}"#)).await.unwrap();
    websocket.send(Message::text(serde_json::json!({"text": format!("{} ", "x".repeat(130))}).to_string())).await.unwrap();
    websocket.send(Message::text(r#"{"text":""}"#)).await.unwrap();
    let audio: serde_json::Value = serde_json::from_str(websocket.next().await.unwrap().unwrap().to_text().unwrap()).unwrap();
    assert!(base64::engine::general_purpose::STANDARD.decode(audio["audio"].as_str().unwrap()).unwrap().len() > 1_000);
    assert_eq!(audio["isFinal"], false);
    let final_message: serde_json::Value = serde_json::from_str(websocket.next().await.unwrap().unwrap().to_text().unwrap()).unwrap();
    assert_eq!(final_message["isFinal"], true);
    websocket.close(None).await.unwrap();

    server.stop().await.unwrap();
    engine.stop().unwrap();
}
