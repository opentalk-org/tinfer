import asyncio
from time import perf_counter

import grpc

from config import grpc_address, model_id, sample_rate, voice_id
from grpc_support import styletts_pb2, styletts_pb2_grpc


async def measure_latency(stub, index: int) -> float:
    request = styletts_pb2.SynthesizeRequest(
        text="To jest pomiar opóźnienia przez gRPC.",
        config=styletts_pb2.SynthesisConfig(model_id=model_id, voice_id=voice_id, sample_rate_hz=sample_rate),
    )
    start = perf_counter()
    first_audio = None
    async for response in stub.SynthesizeStream(request):
        if first_audio is None and response.audio_data:
            first_audio = perf_counter() - start
    assert first_audio is not None, "expected audio"
    print(f"Request {index}: first audio after {first_audio:.3f}s")
    return first_audio


async def main() -> None:
    async with grpc.aio.insecure_channel(grpc_address) as channel:
        stub = styletts_pb2_grpc.StyleTTSServiceStub(channel)
        latencies = await asyncio.gather(*(measure_latency(stub, index) for index in range(3)))
    print(f"Mean latency: {sum(latencies) / len(latencies):.3f}s")


if __name__ == "__main__":
    asyncio.run(main())
