use futures_util::SinkExt;
use serde_json::{Value, json};
use tinfer_rust::server::ServingState;
use tokio_tungstenite::tungstenite::Message;

use super::support::TestServer;

#[tokio::test]
async fn health_endpoints_report_serving_state() {
    let server = TestServer::start().await;
    let live = server.client.get(server.url("/health/live")).send().await.unwrap();
    assert_eq!(live.status(), 200);
    assert_eq!(live.json::<Value>().await.unwrap(), json!({"live":true,"status":"serving"}));

    let ready = server.client.get(server.url("/health/ready")).send().await.unwrap();
    assert_eq!(ready.status(), 200);
    assert_eq!(ready.json::<Value>().await.unwrap(), json!({"ready":true,"status":"serving"}));
    server.stop().await;
}

#[tokio::test]
async fn shutdown_waits_for_websocket_admissions_to_drain() {
    let server = TestServer::start().await;
    let (mut socket, _) =
        tokio_tungstenite::connect_async(server.ws_url("/v1/text-to-speech/default/stream-input?model_id=stub&output_format=pcm_24000"))
            .await
            .unwrap();
    socket.send(Message::text(r#"{"text":" "}"#)).await.unwrap();
    tokio::time::timeout(std::time::Duration::from_secs(1), async {
        while server.health.active_admissions() != 1 {
            tokio::task::yield_now().await;
        }
    })
    .await
    .unwrap();

    let web = server.server.clone();
    let stopping = tokio::spawn(async move { web.stop().await.unwrap() });
    tokio::time::timeout(std::time::Duration::from_secs(1), async {
        while server.health.state() != ServingState::Draining {
            tokio::task::yield_now().await;
        }
    })
    .await
    .unwrap();
    assert!(!stopping.is_finished());
    socket.close(None).await.unwrap();
    tokio::time::timeout(std::time::Duration::from_secs(1), stopping).await.unwrap().unwrap();
    assert_eq!(server.health.state(), ServingState::Stopped);
    server.stop().await;
}
