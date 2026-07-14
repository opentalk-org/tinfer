use tinfer_rust::{Backend, Config, Device};

#[test]
fn config_is_one_small_models_list() {
    let config: Config = serde_yaml::from_str(
        r#"
models:
  - id: stub
    model: stub
    path: models/stub
    backend: onnx
    device: auto
    max_batch: 8
grpc_address: 127.0.0.1:50051
http_address: 127.0.0.1:8000
"#,
    )
    .unwrap();

    assert_eq!(config.models.len(), 1);
    assert_eq!(config.models[0].backend, Backend::Onnx);
    assert_eq!(config.models[0].device, Device::Auto);
    config.validate().unwrap();
}

#[test]
fn config_rejects_old_or_invalid_shapes() {
    for yaml in [
        "automatic_models: []\nplacements: []",
        "models: [{id: stub, path: x, backend: onnx, device: cpu, max_batch: 1, replicas: 2}]",
        "models: [{id: stub, path: x, backend: onnx, device: cuda, max_batch: 1}]",
        "models: [{id: stub, path: x, backend: onnx, device: cpu, max_batch: 0}]",
    ] {
        let invalid = match serde_yaml::from_str::<Config>(yaml) {
            Ok(config) => config.validate().is_err(),
            Err(_) => true,
        };
        assert!(invalid);
    }
}
