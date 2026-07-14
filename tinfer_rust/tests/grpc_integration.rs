use std::net::{IpAddr, Ipv4Addr, SocketAddr};
use std::path::Path;
use std::sync::Arc;
use std::time::Duration;

use tinfer_rust::server::pb::style_tts_service_client::StyleTtsServiceClient;
use tinfer_rust::server::pb::{
    HealthRequest, IncrementalSynthesizeRequest, ListModelsRequest, ListVoicesRequest, SynthesisConfig, SynthesizeRequest,
    incremental_synthesize_request,
};
use tinfer_rust::server::{GrpcConfig, GrpcServer, HealthState};
use tinfer_rust::{AsyncEngine, Backend, Config, Device, Engine, ModelConfig};
use tonic::Code;
use tonic_health::pb::health_client::HealthClient;
use tonic_health::pb::{HealthCheckRequest, health_check_response};

fn config() -> SynthesisConfig {
    SynthesisConfig { model_id: "stub".into(), voice_id: "default".into(), sample_rate_hz: 24_000, language: "en-us".into() }
}

#[tokio::test]
async fn grpc_surface_and_lifecycle_work_over_loopback() {
    let engine = Engine::new(Config {
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
    .unwrap();
    let health = Arc::new(HealthState::new());
    let server = Arc::new(GrpcServer::new(
        AsyncEngine::new(engine.clone()),
        health.clone(),
        GrpcConfig { address: SocketAddr::new(IpAddr::V4(Ipv4Addr::LOCALHOST), 0), shutdown_grace: Duration::from_millis(100) },
    ));
    let address = server.start().await.unwrap();
    let mut client = StyleTtsServiceClient::connect(format!("http://{address}")).await.unwrap();
    let channel = tonic::transport::Endpoint::from_shared(format!("http://{address}")).unwrap().connect().await.unwrap();
    let mut standard_health = HealthClient::new(channel);

    let health_response = client.health(HealthRequest {}).await.unwrap().into_inner();
    assert!(health_response.ready);
    let standard = standard_health.check(HealthCheckRequest { service: String::new() }).await.unwrap().into_inner();
    assert_eq!(standard.status, health_check_response::ServingStatus::Serving as i32);
    let models = client.list_models(ListModelsRequest {}).await.unwrap().into_inner();
    assert_eq!(models.models[0].model_id, "stub");
    let voices = client.list_voices(ListVoicesRequest { model_id: "stub".into() }).await.unwrap().into_inner();
    assert_eq!(voices.voices[0].voice_id, "default");

    let request = SynthesizeRequest { text: "hello".into(), config: Some(config()) };
    let unary = client.synthesize(request.clone()).await.unwrap().into_inner();
    assert!(!unary.audio_data.is_empty());
    let resampled = client
        .synthesize(SynthesizeRequest {
            text: "hello".into(),
            config: Some(SynthesisConfig { sample_rate_hz: 8_000, language: String::new(), ..config() }),
        })
        .await
        .unwrap()
        .into_inner();
    assert!(!resampled.audio_data.is_empty());
    assert!(resampled.audio_data.len() < unary.audio_data.len() / 2);
    let mut streamed = client.synthesize_stream(request).await.unwrap().into_inner();
    assert!(!streamed.message().await.unwrap().unwrap().audio_data.is_empty());

    let messages = vec![
        IncrementalSynthesizeRequest { content: Some(incremental_synthesize_request::Content::Config(config())) },
        IncrementalSynthesizeRequest { content: Some(incremental_synthesize_request::Content::TextChunk(String::new())) },
        IncrementalSynthesizeRequest { content: Some(incremental_synthesize_request::Content::TextChunk("hello".into())) },
        IncrementalSynthesizeRequest { content: Some(incremental_synthesize_request::Content::ForceSynthesis(Default::default())) },
    ];
    let mut incremental = client.synthesize_incremental(tokio_stream::iter(messages)).await.unwrap().into_inner();
    assert!(!incremental.message().await.unwrap().unwrap().audio_data.is_empty());

    let bad_voice = SynthesizeRequest { text: "hello".into(), config: Some(SynthesisConfig { voice_id: "missing".into(), ..config() }) };
    assert_eq!(client.synthesize(bad_voice).await.unwrap_err().code(), Code::NotFound);

    let bad_order =
        vec![IncrementalSynthesizeRequest { content: Some(incremental_synthesize_request::Content::TextChunk("early".into())) }];
    assert_eq!(client.synthesize_incremental(tokio_stream::iter(bad_order)).await.unwrap_err().code(), Code::FailedPrecondition);
    assert_eq!(health.active_admissions(), 0);

    let (requests, inbound) = tokio::sync::mpsc::channel(2);
    requests.send(IncrementalSynthesizeRequest { content: Some(incremental_synthesize_request::Content::Config(config())) }).await.unwrap();
    let held = client.synthesize_incremental(tokio_stream::wrappers::ReceiverStream::new(inbound)).await.unwrap().into_inner();
    while health.active_admissions() != 1 {
        tokio::task::yield_now().await;
    }
    let stopping = server.clone();
    let stop = tokio::spawn(async move { stopping.stop().await });
    while health.state() != tinfer_rust::server::ServingState::Draining {
        tokio::task::yield_now().await;
    }
    let draining = standard_health.check(HealthCheckRequest { service: String::new() }).await.unwrap().into_inner();
    assert_eq!(draining.status, health_check_response::ServingStatus::NotServing as i32);
    let rejected_during_drain = SynthesizeRequest { text: "hello".into(), config: Some(config()) };
    assert_eq!(client.synthesize(rejected_during_drain).await.unwrap_err().code(), Code::Unavailable);
    drop(requests);
    drop(held);
    stop.await.unwrap().unwrap();
    assert_eq!(health.active_admissions(), 0);
    assert_eq!(client.health(HealthRequest {}).await.unwrap_err().code(), Code::Unavailable);
    engine.stop().unwrap();
}
