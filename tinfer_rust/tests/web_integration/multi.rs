use std::collections::HashSet;
use std::time::Duration;

use futures_util::{SinkExt, StreamExt};
use serde_json::{Value, json};
use tokio_tungstenite::tungstenite::Message;

use super::support::TestServer;

async fn receive_json(socket: &mut tokio_tungstenite::WebSocketStream<tokio_tungstenite::MaybeTlsStream<tokio::net::TcpStream>>) -> Value {
    let message = tokio::time::timeout(Duration::from_secs(2), socket.next()).await.unwrap().unwrap().unwrap();
    serde_json::from_str(message.to_text().unwrap()).unwrap()
}

#[tokio::test]
async fn contexts_are_isolated_and_drained_before_socket_close() {
    let server = TestServer::start().await;
    let (mut socket, _) = tokio_tungstenite::connect_async(server.ws_url(
        "/v1/text-to-speech/default/multi-stream-input?model_id=stub&output_format=pcm_24000",
    ))
    .await
    .unwrap();
    for message in [
        json!({"context_id":"a","text":" "}),
        json!({"context_id":"a","text":"Alpha ","flush":true}),
        json!({"context_id":"b","text":"Beta ","flush":true}),
    ] {
        socket.send(Message::text(message.to_string())).await.unwrap();
    }
    let mut ids = HashSet::new();
    for _ in 0..2 {
        let event = receive_json(&mut socket).await;
        assert_eq!(event["isFinal"], false);
        ids.insert(event["contextId"].as_str().unwrap().to_owned());
    }
    assert_eq!(ids, HashSet::from(["a".to_owned(), "b".to_owned()]));
    socket.send(Message::text(json!({"close_socket":true}).to_string())).await.unwrap();
    let mut finals = HashSet::new();
    for _ in 0..2 {
        let event = receive_json(&mut socket).await;
        assert_eq!(event["isFinal"], true);
        finals.insert(event["contextId"].as_str().unwrap().to_owned());
    }
    assert_eq!(finals, ids);
    server.stop().await;
}

#[tokio::test]
async fn unknown_context_transition_is_a_policy_error() {
    let server = TestServer::start().await;
    let (mut socket, _) = tokio_tungstenite::connect_async(server.ws_url(
        "/v1/text-to-speech/default/multi-stream-input?model_id=stub&output_format=pcm_24000",
    ))
    .await
    .unwrap();
    socket.send(Message::text(r#"{"context_id":"a","text":" "}"#)).await.unwrap();
    socket.send(Message::text(r#"{"context_id":"missing","close_context":true}"#)).await.unwrap();
    assert!(receive_json(&mut socket).await["error"].as_str().unwrap().contains("unknown context"));
    server.stop().await;
}
