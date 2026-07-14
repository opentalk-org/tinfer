use std::path::Path;

use tinfer_rust::{AsyncEngine, Backend, Config, Device, Engine, ModelConfig, StreamParams};

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

#[tokio::test]
async fn async_engine_wraps_normal_engine_without_duplicating_state() {
    let normal = engine();
    let asynchronous = AsyncEngine::new(normal.clone());
    let chunk = asynchronous.generate_full("stub", "default", "async", StreamParams::default()).await.unwrap();
    assert_eq!(chunk.sample_rate, 24_000);
    assert_eq!(asynchronous.get_model_ids().await.unwrap(), vec!["stub"]);
    asynchronous.stop().await.unwrap();
}

#[tokio::test]
async fn async_stream_adapts_blocking_stream_operations() {
    let asynchronous = AsyncEngine::new(engine());
    let stream = asynchronous.create_stream("stub", "default", StreamParams::default()).await.unwrap();
    stream.add_text("stream").await.unwrap();
    stream.force_generate().await.unwrap();
    assert!(!stream.recv().await.unwrap().unwrap().audio.is_empty());
    assert!(stream.recv().await.unwrap().is_none());
    stream.close().await.unwrap();
    asynchronous.stop().await.unwrap();
}
