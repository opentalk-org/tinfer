use serde_json::{Value, json};

use super::support::TestServer;

#[tokio::test]
async fn plain_stream_is_chunked_and_nonempty() {
    let server = TestServer::start().await;
    let response = server
        .client
        .post(server.url("/v1/text-to-speech/default/stream?output_format=pcm_24000"))
        .json(&json!({"text": format!("{} ", "speech ".repeat(50)), "model_id":"stub"}))
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), 200);
    assert!(response.headers().get("content-length").is_none());
    assert_eq!(response.headers()["content-type"], "application/octet-stream");
    assert!(!response.bytes().await.unwrap().is_empty());
    server.stop().await;
}

#[tokio::test]
async fn timestamp_stream_is_newline_delimited_json() {
    let server = TestServer::start().await;
    let response = server
        .client
        .post(server.url("/v1/text-to-speech/default/stream/with-timestamps?output_format=pcm_24000"))
        .json(&json!({"text":"Alpha. Beta.","model_id":"stub"}))
        .send()
        .await
        .unwrap();
    assert_eq!(response.headers()["content-type"], "text/event-stream");
    let body = response.text().await.unwrap();
    let records = body.lines().map(serde_json::from_str::<Value>).collect::<Result<Vec<_>, _>>().unwrap();
    assert!(!records.is_empty());
    assert!(records.iter().all(|record| record["audio_base64"].is_string()));
    server.stop().await;
}

#[tokio::test]
async fn wav_streaming_is_rejected_before_headers() {
    let server = TestServer::start().await;
    let response = server
        .client
        .post(server.url("/v1/text-to-speech/default/stream?output_format=wav_24000"))
        .json(&json!({"text":"Hello","model_id":"stub"}))
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), 422);
    server.stop().await;
}
