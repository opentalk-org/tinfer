from pathlib import Path
import asyncio
import random
import signal
from dotenv import load_dotenv

from tinfer.core.engine import StreamingTTS
from tinfer.core.async_engine import AsyncStreamingTTS
from tinfer.config.engine_config import StreamingTTSConfig
from tinfer.server.websocket import WebSocketServer
from tinfer.server.grpc.server import GRPCServer
import os

base_dir = Path(__file__).parent.parent
script_dir = Path(__file__).parent

def _load_tts_config() -> StreamingTTSConfig:
    path = script_dir / "config.yml"
    if path.exists():
        config = StreamingTTSConfig.from_yaml(path)
        print(f"Loaded config from {path}")
        return config
    print(f"Warning: config.yml not found at {path}")
    print("Using default config")
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


def load_models():
    converted_models_dir = os.getenv("TINFER_MODELS_DIR", str(base_dir / "converted_models"))
    converted_models_dir = Path(converted_models_dir)
    
    if not converted_models_dir.exists():
        print(f"Warning: converted_models directory not found at {converted_models_dir}")
        return StreamingTTS(_load_tts_config()), []
    
    config = _load_tts_config()
    tts = StreamingTTS(config)
    
    model_ids = []
    voice_ids = []
    
    for model_dir in converted_models_dir.iterdir():
        if not model_dir.is_dir():
            continue
        
        model_path = model_dir / "model.pth"
        voices_folder = model_dir / "voices"
        
        if not model_path.exists():
            print(f"Skipping {model_dir.name}: model.pth not found")
            continue
        
        model_id = model_dir.name
        voices_folder_str = str(voices_folder) if voices_folder.exists() else None
        
        print(f"Loading model '{model_id}' from {model_path}")
        if voices_folder_str:
            print(f"Loading voices from {voices_folder_str}")
        
        tts.load_model(model_id, str(model_path), voices_folder=voices_folder_str)
        model_ids.append(model_id)
        voice_ids.append(_random_voice_from_folder(voices_folder))
    
    if not model_ids:
        print("No models found to load")
        return tts, []
    
    print(f"\nWarming up {len(model_ids)} model(s)...")
    tts.warmup(model_ids, voice_ids)
    print("All models ready!")
    
    return tts, model_ids

async def main():
    load_dotenv()

    print("Starting TTS Servers...")
    
    tts, model_ids = load_models()
    if not model_ids:
        print("No models loaded. Exiting.")
        return
    
    async_tts = AsyncStreamingTTS(tts)

    host = os.getenv("TINFER_HOST", "0.0.0.0")
    websocket_port = int(os.getenv("TINFER_WEBSOCKET_PORT", "8000"))
    grpc_port = int(os.getenv("TINFER_GRPC_PORT", "50051"))
    use_websocket = os.getenv("USE_WEBSOCKET", "true").lower() in ("true", "1", "yes")
    use_grpc = os.getenv("USE_GRPC", "true").lower() in ("true", "1", "yes")
    
    servers = []
    if use_websocket:
        servers.append((WebSocketServer(async_tts, host=host, port=websocket_port), f"WebSocket {host}:{websocket_port}"))
    if use_grpc:
        servers.append((GRPCServer(async_tts, port=grpc_port), f"gRPC {host}:{grpc_port}"))
    
    shutdown_event = asyncio.Event()
    
    def signal_handler(sig, frame):
        print("\nShutting down servers...")
        try:
            loop = asyncio.get_running_loop()
            loop.call_soon_threadsafe(shutdown_event.set)
        except RuntimeError:
            shutdown_event.set()
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    for _, label in servers:
        print(f"{label}...")
    print("Press Ctrl+C to stop the servers")
    
    tasks = [asyncio.create_task(s.serve()) for s, _ in servers]
    
    try:
        await shutdown_event.wait()
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        for s, _ in servers:
            await s.stop(grace_period=0)
        for t in tasks:
            t.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        async_tts.stop()
        print("Servers stopped")

if __name__ == "__main__":
    asyncio.run(main())
