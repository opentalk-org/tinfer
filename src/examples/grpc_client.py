import asyncio
import grpc
import numpy as np
import soundfile as sf
from pathlib import Path

from tinfer.server.grpc import styletts_pb2, styletts_pb2_grpc

from utils import base_dir, model_name, voice_id

import random
import time
random.seed(int(time.time() * 1_000_000))

packs = [
    ("agnieszka", "66a4ecf82e2a7ae68b14add9_7.97_4.27"),
    ("magda", "magda_001"),
    ("olam", "any"),
]



model_name, voice_id = random.choice(packs)

print(f"Using model: {model_name} with voice: {voice_id}")

output_dir = base_dir / "output_wavs"
output_dir.mkdir(parents=True, exist_ok=True)

SERVER_ADDRESS = "localhost:50051"
SAMPLE_RATE = 24000

def bytes_to_audio(audio_bytes: bytes, sample_rate: int) -> np.ndarray:
    audio_int16 = np.frombuffer(audio_bytes, dtype=np.int16)
    audio_float = audio_int16.astype(np.float32) / 32767.0
    return audio_float

def save_audio(audio_data: bytes, output_path: Path, sample_rate: int):
    audio_array = bytes_to_audio(audio_data, sample_rate)
    sf.write(str(output_path), audio_array, sample_rate)
    print(f"Saved audio to {output_path}")

async def example_synthesize_unary():
    print("\n=== Example 1: Synthesize (Unary RPC) ===")
    print("Standard request/response for short phrases")
    
    async with grpc.aio.insecure_channel(SERVER_ADDRESS) as channel:
        stub = styletts_pb2_grpc.StyleTTSServiceStub(channel)
        
        request = styletts_pb2.SynthesizeRequest(
            text="Dziś jest 2024-04-18. W kalendarzu pojawiło się wydarzenie: \"Spotkanie z Łukaszēm o 13:45\". Czy potrafisz przeczytać znaki nietypowe, takie jak: ñ, ü, ç, æ, ß oraz symbole: ©, ™, €, £? W poniedziałek 01/05/2023 – święto państwowe. Użytkownik Piotr napisał: 'Zażółć gęślą jaźń! 🎉'. Sprawdź, czy TTS wypowie poprawnie daty, symbole i znaki diakrytyczne.",
            config=styletts_pb2.SynthesisConfig(
                model_id=model_name,
                voice_id=voice_id,
                sample_rate_hz=SAMPLE_RATE
            )
        )
        
        print("Sending request...")
        response = await stub.Synthesize(request)
        
        print(f"Received audio: {len(response.audio_data)} bytes")
        print(f"Received {len(response.alignments)} word alignments")
        
        if response.alignments:
            print("\nFirst 5 alignments:")
            for i, alignment in enumerate(response.alignments[:5]):
                print(f"  {i+1}. '{alignment.word}' - {alignment.start_ms}ms to {alignment.end_ms}ms")
        
        print(f"total text: {" ".join([al.word for al in response.alignments])}")

        output_path = output_dir / f"grpc_unary_{model_name}_{voice_id}.wav"
        save_audio(response.audio_data, output_path, SAMPLE_RATE)

async def example_synthesize_stream():
    print("\n=== Example 2: SynthesizeStream (Server Streaming) ===")
    print("Send complete text, receive audio chunks as they're generated")
    
    async with grpc.aio.insecure_channel(SERVER_ADDRESS) as channel:
        stub = styletts_pb2_grpc.StyleTTSServiceStub(channel)
        
        request = styletts_pb2.SynthesizeRequest(
            text="To jest przykład strumieniowania serwera. Pełny tekst jest wysyłany jednocześnie, ale audio jest przesyłane z powrotem w fragmentach w miarę generowania. Pozwala to na niższe opóźnienie, ponieważ możesz zacząć odtwarzać audio przed zakończeniem całej syntezy.",
            config=styletts_pb2.SynthesisConfig(
                model_id=model_name,
                voice_id=voice_id,
                sample_rate_hz=SAMPLE_RATE
            )
        )
        
        print("Sending request, receiving stream...")
        audio_chunks = []
        chunk_count = 0
        
        async for response in stub.SynthesizeStream(request):
            chunk_count += 1
            print(f"Received chunk {chunk_count}: {len(response.audio_data)} bytes, {len(response.alignments)} alignments")
            audio_chunks.append(response.audio_data)
        
        print(f"\nTotal chunks received: {chunk_count}")
        
        if audio_chunks:
            combined_audio = b"".join(audio_chunks)
            output_path = output_dir / f"grpc_stream_{model_name}_{voice_id}.wav"
            save_audio(combined_audio, output_path, SAMPLE_RATE)

async def example_synthesize_incremental_basic():
    print("\n=== Example 3: SynthesizeIncremental (Bidirectional Streaming) - Basic ===")
    print("Send text incrementally, receive audio chunks as they're generated")
    
    async with grpc.aio.insecure_channel(SERVER_ADDRESS) as channel:
        stub = styletts_pb2_grpc.StyleTTSServiceStub(channel)
        
        async def request_iterator():
            yield styletts_pb2.IncrementalSynthesizeRequest(
                config=styletts_pb2.SynthesisConfig(
                    model_id=model_name,
                    voice_id=voice_id,
                    sample_rate_hz=SAMPLE_RATE
                )
            )
            
            text_chunks = [
                "To jest pierwszy fragment tekstu. ",
                "To jest drugi fragment. ",
                "To jest trzeci i ostatni fragment tekstu w tym przykładzie."
            ]
            
            for chunk in text_chunks:
                print(f"Sending text chunk: '{chunk[:30]}...'")
                yield styletts_pb2.IncrementalSynthesizeRequest(
                    text_chunk=chunk
                )
                await asyncio.sleep(0.1)
            
            await asyncio.sleep(0.5)
            yield styletts_pb2.IncrementalSynthesizeRequest(
                force_synthesis=styletts_pb2.ForceSynthesis()
            )
        
        print("Starting bidirectional stream...")
        audio_chunks = []
        chunk_count = 0
        
        async for response in stub.SynthesizeIncremental(request_iterator()):
            chunk_count += 1
            print(f"Received audio chunk {chunk_count}: {len(response.audio_data)} bytes")
            audio_chunks.append(response.audio_data)
        
        print(f"\nTotal chunks received: {chunk_count}")
        
        if audio_chunks:
            combined_audio = b"".join(audio_chunks)
            output_path = output_dir / "grpc_incremental_basic.wav"
            save_audio(combined_audio, output_path, SAMPLE_RATE)

async def example_synthesize_incremental_force():
    print("\n=== Example 4: SynthesizeIncremental with ForceSynthesis ===")
    print("Demonstrates forcing synthesis before all text is sent")
    
    async with grpc.aio.insecure_channel(SERVER_ADDRESS) as channel:
        stub = styletts_pb2_grpc.StyleTTSServiceStub(channel)
        
        async def request_iterator():
            yield styletts_pb2.IncrementalSynthesizeRequest(
                config=styletts_pb2.SynthesisConfig(
                    model_id=model_name,
                    voice_id=voice_id,
                    sample_rate_hz=SAMPLE_RATE
                )
            )
            
            yield styletts_pb2.IncrementalSynthesizeRequest(
                text_chunk="To jest pierwsza część tekstu. "
            )
            
            await asyncio.sleep(0.2)
            
            print("Sending ForceSynthesis request...")
            yield styletts_pb2.IncrementalSynthesizeRequest(
                force_synthesis=styletts_pb2.ForceSynthesis()
            )
            
            await asyncio.sleep(0.5)
            
            yield styletts_pb2.IncrementalSynthesizeRequest(
                text_chunk="To jest dodatkowy tekst wysłany po wymuszeniu syntezy. "
            )
            
            await asyncio.sleep(0.2)
            
            print("Sending another ForceSynthesis request...")
            yield styletts_pb2.IncrementalSynthesizeRequest(
                force_synthesis=styletts_pb2.ForceSynthesis()
            )
        
        print("Starting bidirectional stream with force synthesis...")
        audio_chunks = []
        chunk_count = 0
        
        async for response in stub.SynthesizeIncremental(request_iterator()):
            chunk_count += 1
            print(f"Received audio chunk {chunk_count}: {len(response.audio_data)} bytes")
            audio_chunks.append(response.audio_data)
        
        print(f"\nTotal chunks received: {chunk_count}")
        
        if audio_chunks:
            combined_audio = b"".join(audio_chunks)
            output_path = output_dir / "grpc_incremental_force.wav"
            save_audio(combined_audio, output_path, SAMPLE_RATE)

async def example_synthesize_incremental_cancel():
    print("\n=== Example 5: SynthesizeIncremental with CancelSynthesis ===")
    print("Demonstrates canceling synthesis mid-stream")
    
    async with grpc.aio.insecure_channel(SERVER_ADDRESS) as channel:
        stub = styletts_pb2_grpc.StyleTTSServiceStub(channel)
        
        async def request_iterator():
            yield styletts_pb2.IncrementalSynthesizeRequest(
                config=styletts_pb2.SynthesisConfig(
                    model_id=model_name,
                    voice_id=voice_id,
                    sample_rate_hz=SAMPLE_RATE
                )
            )
            
            yield styletts_pb2.IncrementalSynthesizeRequest(
                text_chunk="To jest długi tekst, po nim zostanie wysłany cancel"
            )
            
            yield styletts_pb2.IncrementalSynthesizeRequest(
                text_chunk="To jest więcej tekstu, który nie powinien być w pełni przetworzony. "
            )
            
            await asyncio.sleep(0.3)
            
            print("Sending CancelSynthesis request...")
            yield styletts_pb2.IncrementalSynthesizeRequest(
                cancel=styletts_pb2.CancelSynthesis()
            )
            
            await asyncio.sleep(0.5)
            
            print("Sending text after cancel (stream should continue)...")
            yield styletts_pb2.IncrementalSynthesizeRequest(
                text_chunk="Ten tekst jest wysyłany po anulowaniu. "
            )
            
            await asyncio.sleep(0.5)
            yield styletts_pb2.IncrementalSynthesizeRequest(
                force_synthesis=styletts_pb2.ForceSynthesis()
            )
        
        print("Starting bidirectional stream with cancel...")
        audio_chunks = []
        chunk_count = 0
        
        async for response in stub.SynthesizeIncremental(request_iterator()):
            chunk_count += 1
            print(f"Received audio chunk {chunk_count}: {len(response.audio_data)} bytes")
            audio_chunks.append(response.audio_data)
        
        print(f"\nTotal chunks received: {chunk_count}")

        print(f"total text: {" ".join([al.word for al in response.alignments])}")
        
        if audio_chunks:
            combined_audio = b"".join(audio_chunks)
            output_path = output_dir / "grpc_incremental_cancel.wav"
            save_audio(combined_audio, output_path, SAMPLE_RATE)

async def example_synthesize_incremental_complete():
    print("\n=== Example 6: SynthesizeIncremental - Complete Workflow ===")
    print("Demonstrates a complete incremental synthesis workflow")
    
    async with grpc.aio.insecure_channel(SERVER_ADDRESS) as channel:
        stub = styletts_pb2_grpc.StyleTTSServiceStub(channel)
        
        async def request_iterator():
            yield styletts_pb2.IncrementalSynthesizeRequest(
                config=styletts_pb2.SynthesisConfig(
                    model_id=model_name,
                    voice_id=voice_id,
                    sample_rate_hz=SAMPLE_RATE
                )
            )
            
            text_parts = [
                "Witamy w kompletnym przykładzie przyrostowej syntezy. ",
                "To pokazuje, jak możesz wysyłać tekst w wielu fragmentach. ",
                "Każdy fragment może być przetwarzany w miarę przybywania. ",
                "Możesz wymusić syntezę w dowolnym momencie. ",
                "Lub pozwolić systemowi automatycznie uruchomić syntezę. ",
                "To jest ostatni fragment tekstu."
            ]
            
            for i, part in enumerate(text_parts):
                print(f"Sending part {i+1}/{len(text_parts)}: '{part[:40]}...'")
                yield styletts_pb2.IncrementalSynthesizeRequest(
                    text_chunk=part
                )
                await asyncio.sleep(0.15)
            
            await asyncio.sleep(0.5)
            print("Forcing final synthesis...")
            yield styletts_pb2.IncrementalSynthesizeRequest(
                force_synthesis=styletts_pb2.ForceSynthesis()
            )
        
        print("Starting complete workflow...")
        audio_chunks = []
        chunk_count = 0
        
        async for response in stub.SynthesizeIncremental(request_iterator()):
            chunk_count += 1
            print(f"Received audio chunk {chunk_count}: {len(response.audio_data)} bytes")
            if response.alignments:
                print(f"  Alignments: {len(response.alignments)} words")
            audio_chunks.append(response.audio_data)
        
        print(f"\nTotal chunks received: {chunk_count}")
        
        if audio_chunks:
            combined_audio = b"".join(audio_chunks)
            output_path = output_dir / "grpc_incremental_complete.wav"
            save_audio(combined_audio, output_path, SAMPLE_RATE)

async def main():
    print("=" * 70)
    print("gRPC TTS Client Examples")
    print("=" * 70)
    print(f"Connecting to server at {SERVER_ADDRESS}")
    print("Make sure the gRPC server is running (run grpc_server.py first)")
    print("=" * 70)
    
    try:
        await example_synthesize_unary()
        await example_synthesize_stream()
        await example_synthesize_incremental_basic()
        await example_synthesize_incremental_force()
        await example_synthesize_incremental_cancel()
        await example_synthesize_incremental_complete()
        
        print("\n" + "=" * 70)
        print("All examples completed!")
        print(f"Output files saved to: {output_dir}")
        print("=" * 70)
    except grpc.RpcError as e:
        print(f"\nError connecting to server: {e.code()} - {e.details()}")
        print("Make sure the gRPC server is running on port 50051")
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())




