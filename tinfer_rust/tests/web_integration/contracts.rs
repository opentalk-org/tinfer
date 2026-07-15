use serde_json::{Value, json};

use super::support::TestServer;

#[tokio::test]
async fn missing_text_has_validation_issue_shape() {
    let server = TestServer::start().await;
    let response = server.client.post(server.url("/v1/text-to-speech/default")).json(&json!({})).send().await.unwrap();
    assert_eq!(response.status(), 422);
    assert_eq!(
        response.json::<Value>().await.unwrap(),
        json!({"detail":[{"loc":["body","text"],"msg":"Field required","type":"missing"}]})
    );
    server.stop().await;
}

#[tokio::test]
async fn unknown_and_invalid_query_values_are_rejected() {
    let server = TestServer::start().await;
    for path in [
        "/v1/text-to-speech/default?unknown=1",
        "/v1/text-to-speech/default?output_format=pcm-whatever-24000",
        "/v1/text-to-speech/default?optimize_streaming_latency=5",
    ] {
        let response = server.client.post(server.url(path)).json(&json!({"text":"Hello"})).send().await.unwrap();
        assert_eq!(response.status(), 422, "{path}");
    }
    server.stop().await;
}

#[tokio::test]
async fn websocket_contract_is_checked_before_upgrade() {
    let server = TestServer::start().await;
    for path in [
        "/v1/text-to-speech/default/stream-input?model_id=missing",
        "/v1/text-to-speech/missing/stream-input?model_id=stub",
        "/v1/text-to-speech/default/stream-input?model_id=stub&output_format=pcm_32000",
        "/v1/text-to-speech/default/stream-input?model_id=stub&inactivity_timeout=181",
    ] {
        let error = tokio_tungstenite::connect_async(server.ws_url(path)).await.unwrap_err();
        let status = match error {
            tokio_tungstenite::tungstenite::Error::Http(response) => response.status(),
            other => panic!("expected HTTP handshake error, got {other}"),
        };
        assert!(matches!(status.as_u16(), 404 | 422), "{path}: {status}");
    }
    server.stop().await;
}
