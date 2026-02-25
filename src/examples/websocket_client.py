import asyncio
import json
import base64
import numpy as np
import soundfile as sf
import websockets

from pathlib import Path
from config import base_dir, model_id, voice_id

output_dir = base_dir / "output_wavs"
output_dir.mkdir(parents=True, exist_ok=True)

SERVER_ADDRESS = "ws://localhost:8000"
OUTPUT_FORMAT = "pcm_24000"
SAMPLE_RATE = 24000
WS_MAX_FRAME_SIZE = 10 * 1024 * 1024

def base64_to_audio(audio_base64: str, sample_rate: int) -> np.ndarray:
    audio_bytes = base64.b64decode(audio_base64)
    audio_int16 = np.frombuffer(audio_bytes, dtype=np.int16)
    audio_float = audio_int16.astype(np.float32) / 32767.0
    return audio_float

def save_audio(audio_data: list[str], output_path: Path, sample_rate: int):
    combined_audio = np.concatenate([base64_to_audio(chunk, sample_rate) for chunk in audio_data])
    sf.write(str(output_path), combined_audio, sample_rate)
    print(f"Saved audio to {output_path}")

def config_message(include_text: str | None = None) -> dict:
    out = {
        "voice_settings": {},
        "generation_config": {"chunk_length_schedule": [120, 160, 250, 290]},
    }
    if include_text is not None:
        out["text"] = include_text
    return out


async def example_simple_synthesis():
    print("\n=== Example 1: Simple Synthesis ===")
    print("Send config with optional text, then close signal; receive audio until isFinal")
    uri = f"{SERVER_ADDRESS}/v1/text-to-speech/{voice_id}/stream-input?model_id={model_id}&output_format={OUTPUT_FORMAT}&sync_alignment=false"
    async with websockets.connect(uri, max_size=WS_MAX_FRAME_SIZE) as websocket:
        initial_message = config_message(
            "To jest prosty przykład żądania w celu przetestowania syntezy mowy przez websocket. "
            "Wysyłamy pełny, stosunkowo długi tekst za jednym żądaniem, aby zobaczyć, jak system radzi sobie z przetwarzaniem większych fragmentów wejścia naraz. "
            "System powinien podzielić dźwięk na porcje i przekazać go stopniowo w odpowiedzi. "
            "Otrzymując całość wypowiedzi, możemy zbadać, czy generowane fragmenty są zgodne z kolejnością tekstu, a ich jakość jest spójna od początku do końca. "
            "Ten przykładowy tekst obejmuje kilka różnych zdań, wykorzystuje przecinki, kropki, a nawet pytania, aby sprawdzić, jak model radzi sobie z różnymi strukturami składniowymi. "
            "Dzięki temu testowi będzie można ocenić działanie generowania mowy, poprawność akcentowania oraz płynność czytania w języku polskim. "
            "Upewniamy się również, że wszystko działa stabilnie podczas obsługi rozbudowanego i wymagającego tekstu wejściowego."
        )
        print("Sending config + text...")
        await websocket.send(json.dumps(initial_message))
        await websocket.send(json.dumps({"text": ""}))
        audio_chunks = []
        chunk_count = 0
        alignment_chars = []
        async for message in websocket:
            data = json.loads(message)
            if "error" in data:
                print(f"Error: {data['error']}")
                break
            if data.get("isFinal"):
                break
            if "audio" in data:
                chunk_count += 1
                audio_chunks.append(data["audio"])
                alignment_count = 0
                if "normalizedAlignment" in data and "chars" in data["normalizedAlignment"]:
                    chars = data["normalizedAlignment"]["chars"]
                    alignment_count = len(chars)
                    alignment_chars.extend(chars)
                print(f"Received chunk {chunk_count}: {len(data['audio'])} bytes (base64), {alignment_count} alignments")
        print(f"\nTotal chunks received: {chunk_count}")
        if alignment_chars:
            concat_alignment = " ".join(str(c) for c in alignment_chars)
            print(f"Concat alignment: {concat_alignment!r}")
        if audio_chunks:
            output_path = output_dir / "websocket_simple.wav"
            save_audio(audio_chunks, output_path, SAMPLE_RATE)

async def example_incremental_basic():
    print("\n=== Example 2: Incremental Synthesis - Basic ===")
    print("Send config, then text incrementally, then close; receive until isFinal")
    uri = f"{SERVER_ADDRESS}/v1/text-to-speech/{voice_id}/stream-input?model_id={model_id}&output_format={OUTPUT_FORMAT}"
    async with websockets.connect(uri, max_size=WS_MAX_FRAME_SIZE) as websocket:
        print("Sending config...")
        await websocket.send(json.dumps(config_message()))
        text_chunks = [
            "To jest pierwszy fragment tekstu. ",
            "To jest drugi fragment. ",
            "To jest trzeci i ostatni fragment tekstu w tym przykładzie.",
        ]
        audio_chunks = []
        chunk_count = 0
        all_text_sent = asyncio.Event()

        async def send_text():
            for chunk in text_chunks:
                print(f"Sending text chunk: '{chunk[:30]}...'")
                await websocket.send(json.dumps({"text": chunk, "try_trigger_generation": False}))
                await asyncio.sleep(0.1)
            await websocket.send(json.dumps({"text": ""}))
            all_text_sent.set()

        async def receive_audio():
            nonlocal chunk_count
            async for message in websocket:
                data = json.loads(message)
                if "error" in data:
                    print(f"Error: {data['error']}")
                    return
                if data.get("isFinal"):
                    return
                if "audio" in data:
                    chunk_count += 1
                    audio_chunks.append(data["audio"])
                    print(f"Received audio chunk {chunk_count}: {len(data['audio'])} bytes (base64)")

        await asyncio.gather(send_text(), receive_audio())
        print(f"\nTotal chunks received: {chunk_count}")
        if audio_chunks:
            output_path = output_dir / "websocket_incremental_basic.wav"
            save_audio(audio_chunks, output_path, SAMPLE_RATE)

async def example_incremental_force():
    print("\n=== Example 3: Incremental Synthesis with Force Generation ===")
    print("Demonstrates forcing synthesis before all text is sent")
    uri = f"{SERVER_ADDRESS}/v1/text-to-speech/{voice_id}/stream-input?model_id={model_id}&output_format={OUTPUT_FORMAT}"
    async with websockets.connect(uri, max_size=WS_MAX_FRAME_SIZE) as websocket:
        print("Sending config...")
        await websocket.send(json.dumps(config_message()))
        audio_chunks = []
        chunk_count = 0
        all_text_sent = asyncio.Event()

        async def send_text():
            await websocket.send(json.dumps({
                "text": "To jest pierwsza część tekstu. ",
                "try_trigger_generation": False,
            }))
            await asyncio.sleep(0.2)
            print("Sending text with force generation...")
            await websocket.send(json.dumps({
                "text": "To jest więcej tekstu. ",
                "try_trigger_generation": True,
            }))
            await asyncio.sleep(0.5)
            await websocket.send(json.dumps({
                "text": "To jest dodatkowy tekst wysłany po wymuszeniu syntezy. ",
                "try_trigger_generation": False,
            }))
            await asyncio.sleep(0.2)
            print("Sending another force generation...")
            await websocket.send(json.dumps({
                "text": "Ostatni fragment. ",
                "try_trigger_generation": True,
            }))
            await websocket.send(json.dumps({"text": ""}))
            all_text_sent.set()

        async def receive_audio():
            nonlocal chunk_count
            async for message in websocket:
                data = json.loads(message)
                if "error" in data:
                    print(f"Error: {data['error']}")
                    return
                if data.get("isFinal"):
                    return
                if "audio" in data:
                    chunk_count += 1
                    audio_chunks.append(data["audio"])
                    print(f"Received audio chunk {chunk_count}: {len(data['audio'])} bytes (base64)")

        await asyncio.gather(send_text(), receive_audio())
        print(f"\nTotal chunks received: {chunk_count}")
        if audio_chunks:
            output_path = output_dir / "websocket_incremental_force.wav"
            save_audio(audio_chunks, output_path, SAMPLE_RATE)

async def example_with_alignments():
    print("\n=== Example 4: Synthesis with Alignments ===")
    print("Demonstrates receiving character alignments (sync_alignment=true)")
    uri = f"{SERVER_ADDRESS}/v1/text-to-speech/{voice_id}/stream-input?model_id={model_id}&output_format={OUTPUT_FORMAT}&sync_alignment=true"
    async with websockets.connect(uri, max_size=WS_MAX_FRAME_SIZE) as websocket:
        initial_message = config_message(
            "To jest przykład z wyrównaniami. Każde słowo ma przypisany czas."
        )
        print("Sending config + text with alignment enabled...")
        await websocket.send(json.dumps(initial_message))
        await websocket.send(json.dumps({"text": ""}))
        audio_chunks = []
        chunk_count = 0
        all_alignments = []
        async for message in websocket:
            data = json.loads(message)
            if "error" in data:
                print(f"Error: {data['error']}")
                break
            if data.get("isFinal"):
                break
            if "audio" in data:
                chunk_count += 1
                audio_chunks.append(data["audio"])
                if "normalizedAlignment" in data:
                    alignment = data["normalizedAlignment"]
                    if "chars" in alignment:
                        all_alignments.extend(alignment["chars"])
                        print(f"Received chunk {chunk_count}: {len(alignment['chars'])} characters aligned")
        print(f"\nTotal chunks received: {chunk_count}")
        print(f"Total aligned characters: {len(all_alignments)}")
        if all_alignments:
            concat_alignment = "".join(str(c) for c in all_alignments)
            print(f"Concat alignment: {concat_alignment!r}")
            print("\nFirst 10 aligned characters:")
            for i, char in enumerate(all_alignments[:10]):
                print(f"  {i+1}. '{char}'")
        if audio_chunks:
            output_path = output_dir / "websocket_alignments.wav"
            save_audio(audio_chunks, output_path, SAMPLE_RATE)

async def example_complete_workflow():
    print("\n=== Example 5: Complete Workflow ===")
    print("Demonstrates a complete incremental synthesis workflow")
    uri = f"{SERVER_ADDRESS}/v1/text-to-speech/{voice_id}/stream-input?model_id={model_id}&output_format={OUTPUT_FORMAT}"
    async with websockets.connect(uri, max_size=WS_MAX_FRAME_SIZE) as websocket:
        print("Sending config...")
        await websocket.send(json.dumps(config_message()))
        text_parts = [
            "Witamy w kompletnym przykładzie przyrostowej syntezy. ",
            "To pokazuje, jak możesz wysyłać tekst w wielu fragmentach. ",
            "Każdy fragment może być przetwarzany w miarę przybywania. ",
            "Możesz wymusić syntezę w dowolnym momencie. ",
            "Lub pozwolić systemowi automatycznie uruchomić syntezę. ",
            "To jest ostatni fragment tekstu.",
        ]
        audio_chunks = []
        chunk_count = 0
        all_text_sent = asyncio.Event()

        async def send_text():
            for i, part in enumerate(text_parts):
                print(f"Sending part {i+1}/{len(text_parts)}: '{part[:40]}...'")
                await websocket.send(json.dumps({"text": part, "try_trigger_generation": False}))
                await asyncio.sleep(0.15)
            await websocket.send(json.dumps({"text": ""}))
            all_text_sent.set()

        async def receive_audio():
            nonlocal chunk_count
            async for message in websocket:
                data = json.loads(message)
                if "error" in data:
                    print(f"Error: {data['error']}")
                    return
                if data.get("isFinal"):
                    return
                if "audio" in data:
                    chunk_count += 1
                    audio_chunks.append(data["audio"])
                    alignment_count = 0
                    if "normalizedAlignment" in data and "chars" in data["normalizedAlignment"]:
                        alignment_count = len(data["normalizedAlignment"]["chars"])
                    print(f"Received audio chunk {chunk_count}: {len(data['audio'])} bytes (base64), {alignment_count} alignments")

        await asyncio.gather(send_text(), receive_audio())
        print(f"\nTotal chunks received: {chunk_count}")
        if audio_chunks:
            output_path = output_dir / "websocket_complete.wav"
            save_audio(audio_chunks, output_path, SAMPLE_RATE)

async def main():
    print("=" * 70)
    print("WebSocket TTS Client Examples")
    print("=" * 70)
    print(f"Connecting to server at {SERVER_ADDRESS}")
    print("Make sure the WebSocket server is running (run websocket_server.py first)")
    print("=" * 70)
    
    try:
        await example_simple_synthesis()
        await example_incremental_basic()
        await example_incremental_force()
        await example_with_alignments()
        await example_complete_workflow()
        
        print("\n" + "=" * 70)
        print("All examples completed!")
        print(f"Output files saved to: {output_dir}")
        print("=" * 70)
    except websockets.exceptions.ConnectionClosed as e:
        print(f"\nConnection closed: {e}")
        print("Make sure the WebSocket server is running on port 8000")
    except ConnectionRefusedError:
        print(f"\nError: Could not connect to server at {SERVER_ADDRESS}")
        print("Make sure the WebSocket server is running on port 8000")
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())
