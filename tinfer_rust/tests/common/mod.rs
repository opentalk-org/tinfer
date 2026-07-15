use std::net::{IpAddr, Ipv4Addr, SocketAddr};
use std::path::Path;

use tinfer_rust::{AlignmentType, Backend, Config, Device, EngineSettings, GrpcSettings, ModelConfig, SynthesisSettings, WebSettings};

pub fn stub_model() -> ModelConfig {
    ModelConfig {
        id: "stub".into(),
        model: "stub".into(),
        path: Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/artifacts/stub"),
        backend: Backend::Onnx,
        device: Device::Cpu,
        max_batch: 4,
        settings: serde_json::json!({}),
    }
}

pub fn config(models: Vec<ModelConfig>) -> Config {
    let localhost = IpAddr::V4(Ipv4Addr::LOCALHOST);
    Config {
        engine: EngineSettings { engine_timeout_ms: 0, queue_capacity: 64 },
        defaults: SynthesisSettings {
            chunk_length_schedule: vec![120, 160, 250, 290],
            inactivity_timeout_ms: 80,
            alignment_type: AlignmentType::Word,
        },
        grpc: GrpcSettings { enabled: true, address: SocketAddr::new(localhost, 0) },
        web: web_settings(SocketAddr::new(localhost, 0)),
        models,
    }
}

pub fn web_settings(address: SocketAddr) -> WebSettings {
    WebSettings {
        enabled: true,
        websocket_enabled: true,
        address,
        output_format: "mp3_44100_128".into(),
        websocket_inactivity_timeout_seconds: 20,
        sync_alignment: false,
        auto_mode: false,
    }
}
