use tinfer_rust::{AlignmentType, Backend, Config, Device};

const CONFIG: &str = r#"
engine:
  engine_timeout_ms: 80
  queue_capacity: 64
defaults:
  chunk_length_schedule: [200]
  inactivity_timeout_ms: 80
  alignment_type: word
grpc:
  enabled: true
  address: 127.0.0.1:50051
web:
  enabled: true
  websocket_enabled: true
  address: 127.0.0.1:8000
  output_format: mp3_44100_128
  websocket_inactivity_timeout_seconds: 20
  sync_alignment: false
  auto_mode: false
models:
  - id: stub
    model: stub
    path: models/stub
    backend: onnx
    device: auto
    max_batch: 8
    settings:
      characters_per_call: 32
"#;

#[test]
fn required_settings_deserialize_from_one_schema() {
    let config: Config = serde_yaml::from_str(CONFIG).unwrap();

    assert_eq!(config.engine.engine_timeout_ms, 80);
    assert_eq!(config.defaults.chunk_length_schedule, [200]);
    assert_eq!(config.defaults.inactivity_timeout_ms, 80);
    assert_eq!(config.defaults.alignment_type, AlignmentType::Word);
    assert!(config.grpc.enabled);
    assert!(config.web.websocket_enabled);
    assert_eq!(config.models[0].backend, Backend::Onnx);
    assert_eq!(config.models[0].device, Device::Auto);
    assert_eq!(config.models[0].settings["characters_per_call"], 32);
    config.validate().unwrap();
}

#[test]
fn every_settings_group_is_required() {
    for group in ["engine", "defaults", "grpc", "web", "models"] {
        let yaml = CONFIG.lines().skip_while(|line| !line.starts_with(&format!("{group}:"))).next().expect("group exists");
        let source = CONFIG.replacen(&format!("{yaml}\n"), "", 1);
        assert!(serde_yaml::from_str::<Config>(&source).is_err(), "{group} must be required");
    }
}

#[test]
fn config_rejects_legacy_or_invalid_shapes() {
    for yaml in [
        "automatic_models: []\nplacements: []",
        &CONFIG.replace("engine_timeout_ms", "timeout_ms"),
        &CONFIG.replace("chunk_length_schedule: [200]", "chunk_length_schedule: []"),
        &CONFIG.replace("max_batch: 8", "max_batch: 0"),
    ] {
        let invalid = serde_yaml::from_str::<Config>(yaml).map_or(true, |config| config.validate().is_err());
        assert!(invalid);
    }
}
