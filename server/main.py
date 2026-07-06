from pathlib import Path
import asyncio
import logging
import random
import signal
from dotenv import load_dotenv

from tinfer.core.engine import StreamingTTS
from tinfer.core.async_engine import AsyncStreamingTTS
from tinfer.config.engine_config import StreamingTTSConfig
from tinfer.server.websocket import WebSocketServer
from tinfer.server.grpc.server import GRPCServer
from tinfer.server.health import HealthState
import os
from tinfer.support.observability import get_logger

from server.observability import setup_observability

base_dir = Path(__file__).parent.parent
script_dir = Path(__file__).parent
log = get_logger(__name__)


def _log_level_from_env() -> int:
    level_name = os.getenv("LOG_LEVEL", "INFO").upper()
    return getattr(logging, level_name, logging.INFO)

def _load_tts_config() -> StreamingTTSConfig:
    path = script_dir / "config.yml"
    if path.exists():
        config = StreamingTTSConfig.from_yaml(path)
        log.info("config_loaded", path=str(path))
        return config
    log.warning("config_missing", path=str(path))
    log.info("config_default_used")
    return StreamingTTSConfig()

def _random_voice_from_folder(voices_folder: Path) -> str:
    if not voices_folder.exists() or not voices_folder.is_dir():
        raise ValueError(f"Voices folder does not exist: {voices_folder}")
    
    names = [
        p.stem for p in voices_folder.iterdir()
        if p.suffix.lower() == ".pth"
    ]

    if not names:
        raise ValueError(f"No voices found in {voices_folder}")
    return random.choice(names)


def load_models(warmup: bool = True):
    converted_models_dir = os.getenv("TINFER_MODELS_DIR", str(base_dir / "converted_models"))
    converted_models_dir = Path(converted_models_dir)
    
    if not converted_models_dir.exists():
        log.warning("converted_models_missing", path=str(converted_models_dir))
        return StreamingTTS(_load_tts_config()), [], []
    
    config = _load_tts_config()
    tts = StreamingTTS(config)

    model_ids = []
    voice_ids = []

    selected = config.models or None
    if selected is not None:
        available = {p.name for p in converted_models_dir.iterdir() if p.is_dir()}
        for name in selected:
            if name not in available:
                log.warning("model_requested_but_missing", model_id=name, path=str(converted_models_dir / name))
        log.info("models_selected", selected=list(selected))

    for model_dir in sorted(converted_models_dir.iterdir()):
        if not model_dir.is_dir():
            continue

        if selected is not None and model_dir.name not in selected:
            log.info("model_not_selected", model_id=model_dir.name)
            continue

        model_path = model_dir / "model.pth"
        voices_folder = model_dir / "voices"

        if not model_path.exists():
            log.warning("model_skipped", model_id=model_dir.name, reason="model_path_missing", path=str(model_path))
            continue

        model_id = model_dir.name
        voices_folder_str = str(voices_folder) if voices_folder.exists() else None
        
        tts.load_model(model_id, str(model_path), voices_folder=voices_folder_str)
        model_ids.append(model_id)
        voice_ids.append(_random_voice_from_folder(voices_folder))
    
    if not model_ids:
        log.warning("models_not_found")
        return tts, [], []
    
    if warmup:
        log.info("models_warmup_started", model_count=len(model_ids))
        tts.warmup(model_ids, voice_ids)
        log.info("models_ready", model_count=len(model_ids))
    
    return tts, model_ids, voice_ids

async def main():
    load_dotenv()
    setup_observability(
        service_name=os.getenv("OTEL_SERVICE_NAME", "tinfer-server"),
        environment=os.getenv("DEPLOYMENT_ENVIRONMENT", os.getenv("ENVIRONMENT", "dev")),
        level=_log_level_from_env(),
    )

    log.info("server_starting")
    health = HealthState()
    
    tts, model_ids, voice_ids = load_models(warmup=False)
    if not model_ids:
        log.warning("server_exiting_no_models")
        return

    async_tts = AsyncStreamingTTS(tts)

    host = os.getenv("TINFER_HOST", "0.0.0.0")
    websocket_port = int(os.getenv("TINFER_WEBSOCKET_PORT", "8000"))
    grpc_port = int(os.getenv("TINFER_GRPC_PORT", "50051"))
    use_websocket = os.getenv("USE_WEBSOCKET", "true").lower() in ("true", "1", "yes")
    use_grpc = os.getenv("USE_GRPC", "true").lower() in ("true", "1", "yes")
    
    servers = []
    if use_websocket:
        servers.append((WebSocketServer(async_tts, host=host, port=websocket_port, health=health), f"WebSocket {host}:{websocket_port}"))
    if use_grpc:
        servers.append((GRPCServer(async_tts, port=grpc_port, health=health), f"gRPC {host}:{grpc_port}"))
    
    shutdown_event = asyncio.Event()
    
    def signal_handler(sig, frame):
        log.info("server_shutdown_signal", signal=str(sig))
        try:
            loop = asyncio.get_running_loop()
            loop.call_soon_threadsafe(shutdown_event.set)
        except RuntimeError:
            shutdown_event.set()
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    tasks = [asyncio.create_task(s.serve()) for s, _ in servers]

    for _, label in servers:
        log.info("server_listener_starting", listener=label)

    try:
        log.info("models_warmup_started", model_count=len(model_ids))
        await tts.async_warmup(model_ids, voice_ids)
        health.mark_warmup_complete()
        log.info("models_ready", model_count=len(model_ids))
        log.info("server_ready")
        await shutdown_event.wait()
    except KeyboardInterrupt:
        log.info("server_shutdown_keyboard_interrupt")
    finally:
        await health.begin_draining()
        log.info("server_draining_started", active_connections=health.active_connections)
        await health.wait_for_no_active_connections()
        log.info("server_draining_finished")
        for s, _ in servers:
            await s.stop(grace_period=0)
        for t in tasks:
            t.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        async_tts.stop()
        await health.mark_stopped()
        log.info("server_stopped")

if __name__ == "__main__":
    import sys

    if "--smoke-test" in sys.argv[1:]:
        from server.smoke_test import main as smoke_test

        sys.exit(smoke_test())
    asyncio.run(main())
