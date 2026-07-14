use std::path::Path;

use tinfer_rust::{Backend, Config, Device, Engine, ModelConfig};

#[test]
fn tensorrt_styletts2_rejects_cpu_before_reading_artifacts() {
    let error = Engine::new(Config {
        models: vec![ModelConfig {
            id: "styletts2".into(),
            model: "styletts2".into(),
            path: Path::new("missing-export").into(),
            backend: Backend::Tensorrt,
            device: Device::Cpu,
            max_batch: 4,
        }],
        ..Config::default()
    })
    .err()
    .expect("TensorRT CPU must fail during model loading");

    assert_eq!(error.to_string(), "StyleTTS2 TensorRT requires a CUDA device");
}
