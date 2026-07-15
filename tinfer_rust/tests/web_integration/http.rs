use serde_json::{Value, json};

use super::support::TestServer;

#[tokio::test]
async fn catalog_envelopes_match_python_contract() {
    let server = TestServer::start().await;
    let models = server.client.get(server.url("/v1/models")).send().await.unwrap().json::<Value>().await.unwrap();
    assert_eq!(models[0]["model_id"], "stub");
    assert_eq!(models[0]["default_language"], "en-us");
    let voices = server.client.get(server.url("/v1/voices")).send().await.unwrap().json::<Value>().await.unwrap();
    assert_eq!(voices["voices"][0]["voice_id"], "default");
    assert_eq!(voices["voices"][0]["model_id"], "stub");
    server.stop().await;
}

#[tokio::test]
async fn unary_audio_and_timing_match_contract() {
    let server = TestServer::start().await;
    let audio = server
        .client
        .post(server.url("/v1/text-to-speech/default?output_format=pcm_24000"))
        .json(&json!({"text":"AB","model_id":"stub","language_code":"en-us"}))
        .send()
        .await
        .unwrap();
    assert_eq!(audio.status(), 200);
    assert_eq!(audio.headers()["content-type"], "application/octet-stream");
    assert!(!audio.bytes().await.unwrap().is_empty());

    let timing = server
        .client
        .post(server.url("/v1/text-to-speech/default/with-timestamps?output_format=pcm_24000"))
        .json(&json!({"text":"AB","model_id":"stub"}))
        .send()
        .await
        .unwrap()
        .json::<Value>()
        .await
        .unwrap();
    assert_eq!(timing["alignment"]["characters"], json!(["A", "B"]));
    assert_eq!(timing["normalized_alignment"]["characters"], json!(["A", "B"]));
    server.stop().await;
}

#[tokio::test]
async fn catalog_and_speech_failures_are_explicit() {
    let server = TestServer::start().await;
    let missing =
        server.client.post(server.url("/v1/text-to-speech/missing")).json(&json!({"text":"Hello","model_id":"stub"})).send().await.unwrap();
    assert_eq!(missing.status(), 404);
    let empty =
        server.client.post(server.url("/v1/text-to-speech/default")).json(&json!({"text":"  ","model_id":"stub"})).send().await.unwrap();
    assert_eq!(empty.status(), 422);
    server.stop().await;
}
