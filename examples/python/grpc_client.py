import asyncio

import grpc

from config import grpc_address, model_id, output_dir, sample_rate, voice_id
from grpc_support import styletts_pb2, styletts_pb2_grpc
from utils import save_audio


async def incremental_requests():
    yield styletts_pb2.IncrementalSynthesizeRequest(config=styletts_pb2.SynthesisConfig(
        model_id=model_id, voice_id=voice_id, sample_rate_hz=sample_rate,
    ))
    for text in ("Pierwszy fragment tekstu. ", "Drugi fragment tekstu."):
        yield styletts_pb2.IncrementalSynthesizeRequest(text_chunk=text)
    yield styletts_pb2.IncrementalSynthesizeRequest(force_synthesis=styletts_pb2.ForceSynthesis())


async def main() -> None:
    request = styletts_pb2.SynthesizeRequest(
        text="To jest przykład klienta gRPC.",
        config=styletts_pb2.SynthesisConfig(model_id=model_id, voice_id=voice_id, sample_rate_hz=sample_rate),
    )
    async with grpc.aio.insecure_channel(grpc_address) as channel:
        stub = styletts_pb2_grpc.StyleTTSServiceStub(channel)
        response = await stub.Synthesize(request)
        save_audio(response.audio_data, output_dir / "grpc_unary.wav")
        chunks = [response.audio_data async for response in stub.SynthesizeStream(request)]
        save_audio(b"".join(chunks), output_dir / "grpc_stream.wav")
        chunks = [response.audio_data async for response in stub.SynthesizeIncremental(incremental_requests())]
        save_audio(b"".join(chunks), output_dir / "grpc_incremental.wav")


if __name__ == "__main__":
    asyncio.run(main())
