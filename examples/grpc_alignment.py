import asyncio

import grpc

from config import grpc_address, model_id, output_dir, sample_rate, voice_id
from grpc_support import styletts_pb2, styletts_pb2_grpc
from utils import save_audio


async def main() -> None:
    request = styletts_pb2.SynthesizeRequest(
        text="To jest przykład wyrównania słów przez gRPC.",
        config=styletts_pb2.SynthesisConfig(model_id=model_id, voice_id=voice_id, sample_rate_hz=sample_rate),
    )
    chunks = []
    elapsed_samples = 0
    async with grpc.aio.insecure_channel(grpc_address) as channel:
        responses = styletts_pb2_grpc.StyleTTSServiceStub(channel).SynthesizeStream(request)
        async for response in responses:
            offset_ms = elapsed_samples * 1000 // sample_rate
            for item in response.alignments:
                print(f"{item.word!r}: {item.start_ms + offset_ms}ms–{item.end_ms + offset_ms}ms")
            chunks.append(response.audio_data)
            elapsed_samples += len(response.audio_data) // 2
    save_audio(b"".join(chunks), output_dir / "grpc_alignment.wav")


if __name__ == "__main__":
    asyncio.run(main())
