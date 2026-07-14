use std::path::Path;
use std::sync::mpsc;
use std::time::Duration;

use tinfer_rust::{AudioChunk, Backend, Config, Device, Engine, ModelConfig, StreamParams};

fn engine() -> Engine {
    Engine::new(Config {
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
    .unwrap()
}

#[test]
fn normal_engine_matches_tinfer_stream_flow() {
    let engine = engine();

    assert_eq!(engine.get_model_ids().unwrap(), vec!["stub"]);
    assert_eq!(engine.get_voice_ids("stub").unwrap(), vec!["default"]);

    let stream = engine.create_stream("stub", "default", StreamParams::default()).unwrap();
    stream.add_text("hello ").unwrap();
    stream.force_generate().unwrap();
    let chunks = stream.collect_audio().unwrap();

    assert_eq!(chunks.len(), 1);
    assert!(!chunks[0].audio.is_empty());
    assert_eq!(chunks[0].text_span, 0..6);
    assert_eq!(chunks[0].chunk_index, 0);
    stream.close().unwrap();
    engine.unload_model("stub").unwrap();
    assert!(engine.get_model_ids().unwrap().is_empty());
    engine.stop().unwrap();
}

#[test]
fn stream_can_generate_more_than_once() {
    let engine = engine();
    let stream = engine.create_stream("stub", "default", StreamParams::default()).unwrap();

    stream.add_text("first ").unwrap();
    stream.force_generate().unwrap();
    assert_eq!(stream.collect_audio().unwrap()[0].text_span, 0..6);

    stream.add_text("second").unwrap();
    stream.force_generate().unwrap();
    assert_eq!(stream.collect_audio().unwrap()[0].text_span, 6..12);

    stream.close().unwrap();
    engine.stop().unwrap();
}

#[test]
fn cancellation_is_delivered_to_the_stream() {
    let engine = engine();
    let stream = engine.create_stream("stub", "default", StreamParams::default()).unwrap();
    stream.add_text("cancel me").unwrap();
    stream.cancel().unwrap();
    assert!(stream.recv().is_err());
    engine.stop().unwrap();
}

#[test]
fn full_generation_merges_chunks() {
    let engine = engine();
    let chunk: AudioChunk = engine.generate_full("stub", "default", "hello", StreamParams::default()).unwrap();
    assert!(!chunk.audio.is_empty());
    assert_eq!(chunk.text_span, 0..5);
    engine.stop().unwrap();
}

#[test]
fn pending_text_runs_when_its_generation_window_expires() {
    let engine = engine();
    let stream =
        engine.create_stream("stub", "default", StreamParams { timeout: Duration::from_millis(5), ..StreamParams::default() }).unwrap();
    stream.add_text("short").unwrap();
    assert!(!stream.collect_audio().unwrap()[0].audio.is_empty());
    engine.stop().unwrap();
}

#[test]
fn forced_long_text_uses_language_sentence_boundaries() {
    let engine = engine();
    let stream =
        engine.create_stream("stub", "default", StreamParams { chunk_length_schedule: vec![18], ..StreamParams::default() }).unwrap();
    stream.add_text("Dr. Smith went home. Another sentence follows.").unwrap();
    stream.force_generate().unwrap();
    let chunks = stream.collect_audio().unwrap();
    assert_eq!(chunks.len(), 3);
    assert_eq!(chunks[0].text_span, 0..21);
    assert_eq!(chunks.last().unwrap().text_span.end, 46);
    engine.stop().unwrap();
}

#[test]
fn stream_control_and_state_are_owned_by_rust() {
    let engine = engine();
    let stream = engine.create_stream("stub", "default", StreamParams::default()).unwrap();
    assert!(stream.get_audio().unwrap().is_empty());
    assert_eq!(stream.get_state().unwrap(), serde_json::json!({}));
    stream.add_text("state").unwrap();
    stream.try_generate().unwrap();
    stream.force_generate().unwrap();
    assert_eq!(stream.collect_audio().unwrap().len(), 1);
    assert_eq!(stream.get_state().unwrap(), serde_json::json!({"text": "state"}));
    stream.close().unwrap();
    assert!(stream.try_generate().is_err());
    engine.stop().unwrap();
}

#[test]
fn unloading_fails_queued_streams_instead_of_leaving_them_waiting() {
    let engine = engine();
    let stream = engine.create_stream("stub", "default", StreamParams::default()).unwrap();
    stream.add_text("queued").unwrap();

    engine.unload_model("stub").unwrap();

    let (tx, rx) = mpsc::sync_channel(1);
    std::thread::spawn(move || tx.send(stream.recv()).unwrap());
    let result = rx.recv_timeout(Duration::from_millis(100)).expect("unload must wake the stream");
    assert_eq!(result.unwrap_err().to_string(), "model unloaded: stub");
    assert!(engine.get_model_ids().unwrap().is_empty());
    engine.stop().unwrap();
}
