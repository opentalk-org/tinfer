import asyncio
import base64
import json
from urllib.parse import quote

import websockets

from config import model_id, output_dir, server_url, voice_id
from utils import save_audio


async def main() -> None:
    address = server_url.replace("http://", "ws://").replace("https://", "wss://")
    uri = f"{address}/v1/text-to-speech/{quote(voice_id, safe='')}/stream-input?model_id={quote(model_id, safe='')}&output_format=pcm_24000"
    chunks = []
    async with websockets.connect(uri, max_size=10 * 1024 * 1024) as websocket:
        await websocket.send(json.dumps({"text": " ", "voice_settings": {}}))
        await websocket.send(json.dumps({"text": "To jest przykład syntezy przez WebSocket. "}))
        await websocket.send(json.dumps({"text": "Drugi fragment tekstu. ", "try_trigger_generation": True}))
        await websocket.send(json.dumps({"text": ""}))
        async for message in websocket:
            response = json.loads(message)
            if "error" in response:
                raise RuntimeError(response["error"])
            if "audio" in response:
                chunks.append(base64.b64decode(response["audio"]))
            if response["isFinal"]:
                break
    save_audio(b"".join(chunks), output_dir / "websocket.wav")


if __name__ == "__main__":
    asyncio.run(main())
