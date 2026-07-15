use std::time::Duration;

use futures_util::{SinkExt, StreamExt};
use serde_json::{Value, json};
use tokio_tungstenite::tungstenite::Message;
use tokio_tungstenite::tungstenite::protocol::frame::coding::CloseCode;

use super::support::TestServer;

async fn receive_json(socket: &mut tokio_tungstenite::WebSocketStream<tokio_tungstenite::MaybeTlsStream<tokio::net::TcpStream>>) -> Value {
    let message = tokio::time::timeout(Duration::from_secs(2), socket.next()).await.unwrap().unwrap().unwrap();
    serde_json::from_str(message.to_text().unwrap()).unwrap()
}

#[tokio::test]
async fn policy_errors_are_reported_before_close() {
    let server = TestServer::start().await;
    let (mut socket, _) =
        tokio_tungstenite::connect_async(server.ws_url("/v1/text-to-speech/default/stream-input?model_id=stub&output_format=pcm_24000"))
            .await
            .unwrap();
    socket.send(Message::text(r#"{"text":"not initialization "}"#)).await.unwrap();
    assert!(receive_json(&mut socket).await["error"].as_str().unwrap().contains("first message"));
    let close = socket.next().await.unwrap().unwrap();
    let Message::Close(Some(frame)) = close else { panic!("expected close frame") };
    assert_eq!(frame.code, CloseCode::Policy);
    server.stop().await;
}

#[tokio::test]
async fn alignment_and_final_messages_match_contract() {
    let server = TestServer::start().await;
    let (mut socket, _) = tokio_tungstenite::connect_async(
        server.ws_url("/v1/text-to-speech/default/stream-input?model_id=stub&output_format=pcm_24000&sync_alignment=true"),
    )
    .await
    .unwrap();
    socket.send(Message::text(r#"{"text":" "}"#)).await.unwrap();
    socket.send(Message::text(r#"{"text":"Hello ","flush":true}"#)).await.unwrap();
    let audio = receive_json(&mut socket).await;
    assert_eq!(audio["isFinal"], false);
    assert!(audio["alignment"]["chars"].is_array());
    socket.send(Message::text(json!({"text":""}).to_string())).await.unwrap();
    while receive_json(&mut socket).await["isFinal"] != true {}
    server.stop().await;
}

#[tokio::test]
async fn inactivity_reports_error_and_closes() {
    let server = TestServer::start().await;
    let (mut socket, _) = tokio_tungstenite::connect_async(
        server.ws_url("/v1/text-to-speech/default/stream-input?model_id=stub&output_format=pcm_24000&inactivity_timeout=1"),
    )
    .await
    .unwrap();
    socket.send(Message::text(r#"{"text":" "}"#)).await.unwrap();
    assert!(receive_json(&mut socket).await["error"].as_str().unwrap().contains("inactivity"));
    server.stop().await;
}

#[tokio::test]
async fn initialization_settings_are_strict_and_immutable() {
    let server = TestServer::start().await;
    let (mut socket, _) =
        tokio_tungstenite::connect_async(server.ws_url("/v1/text-to-speech/default/stream-input?model_id=stub&output_format=pcm_24000"))
            .await
            .unwrap();
    socket
        .send(Message::text(r#"{"text":" ","voice_settings":{"speed":1.1},"generation_config":{"chunk_length_schedule":[50]}}"#))
        .await
        .unwrap();
    socket.send(Message::text(r#"{"text":"Hello ","flush":"yes"}"#)).await.unwrap();
    assert!(receive_json(&mut socket).await["error"].as_str().unwrap().contains("flush must be boolean"));

    let (mut socket, _) =
        tokio_tungstenite::connect_async(server.ws_url("/v1/text-to-speech/default/stream-input?model_id=stub&output_format=pcm_24000"))
            .await
            .unwrap();
    socket.send(Message::text(r#"{"text":" ","voice_settings":{"speed":1.1}}"#)).await.unwrap();
    socket.send(Message::text(r#"{"text":"Hello ","voice_settings":{"speed":1.0}}"#)).await.unwrap();
    assert!(receive_json(&mut socket).await["error"].as_str().unwrap().contains("cannot change"));
    server.stop().await;
}
