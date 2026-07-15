use tinfer_rust::{Config, Engine};

mod common;

fn config() -> Config {
    let mut model = common::stub_model();
    model.id = "voice-a".into();
    common::config(vec![model])
}

#[test]
fn configured_stub_loads_and_calls_native_generate_batch() {
    let engine = Engine::new(config()).unwrap();
    let chunk = engine.generate_full("voice-a", "default", "abc", engine.stream_params()).unwrap();
    assert_eq!(chunk.sample_rate, 24_000);
    assert_eq!(chunk.audio.len(), 123);
    assert!(chunk.audio.iter().any(|sample| *sample != 0.0));
    engine.stop().unwrap();
}

#[test]
fn configured_model_load_failure_is_returned_by_engine_new() {
    let mut config = config();
    config.models[0].path = "missing".into();
    assert!(Engine::new(config).is_err());
}
