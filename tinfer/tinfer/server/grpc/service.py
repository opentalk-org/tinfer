from __future__ import annotations
import asyncio
import numpy as np
import grpc

from tinfer.core.async_engine import AsyncStreamingTTS
from tinfer.core.request import AudioChunk, AlignmentType
from tinfer.utils.alignment_formatter import AlignmentFormatter
from tinfer.utils.audio_encoder import AudioFormat
from typing import AsyncIterator, Any
import asyncio
from . import styletts_pb2
from . import styletts_pb2_grpc
from tinfer.core.request import Alignment
from tinfer.errors import InferenceError

class StyleTTSService(styletts_pb2_grpc.StyleTTSServiceServicer):

    def __init__(self, tts: AsyncStreamingTTS) -> None:
        self.tts = tts
        self.alignment_formatter = AlignmentFormatter()

    def _audio_to_bytes(self, audio: np.ndarray) -> bytes:
        if audio.dtype == np.uint8:
            return audio.tobytes()
        audio_int16 = (audio * 32767.0).astype(np.int16)
        return audio_int16.tobytes()

    def _create_response(self, audio: bytes, alignment: Alignment | None) -> styletts_pb2.SynthesizeResponse:

        response = styletts_pb2.SynthesizeResponse()
        response.audio_data = audio

        if alignment is not None and alignment.type_ != AlignmentType.NONE:
            for item in alignment.items:
                alignment_item = response.alignments.add()
                alignment_item.word = item.item
                alignment_item.start_ms = item.start_ms
                alignment_item.end_ms = item.end_ms

        return response

    def _get_params(self, request: styletts_pb2.SynthesizeRequest | styletts_pb2.IncrementalSynthesizeRequest) -> tuple[str, str, dict[str, Any]]:
        config = request.config
        model_id = config.model_id
        voice_id = config.voice_id
        sample_rate = config.sample_rate_hz
        
        if sample_rate == 8000:
            target_encoding = AudioFormat.PCM_8000
        elif sample_rate == 16000:
            target_encoding = AudioFormat.PCM_16000
        elif sample_rate == 22050:
            target_encoding = AudioFormat.PCM_22050
        elif sample_rate == 24000:
            target_encoding = AudioFormat.PCM_24000
        elif sample_rate == 44100:
            target_encoding = AudioFormat.PCM_44100
        else:
            target_encoding = AudioFormat.PCM_24000
        
        return model_id, voice_id, dict(
            target_sample_rate=sample_rate,
            target_encoding=target_encoding,
        )

    def _chunk_to_response(self, chunk: AudioChunk) -> styletts_pb2.SynthesizeResponse:
        audio_bytes = self._audio_to_bytes(chunk.audio)
        alignments = chunk.alignments
        return self._create_response(audio_bytes, alignments)

    async def Synthesize(self, request: styletts_pb2.SynthesizeRequest, context: grpc.ServicerContext) -> styletts_pb2.SynthesizeResponse:
        model_id, voice_id, params = self._get_params(request)
        text = request.text
        try:
            chunk: AudioChunk = self.tts.generate_full(
                model_id=model_id,
                voice_id=voice_id,
                text=text,
                params=params,
            )
        except InferenceError as e:
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details(str(e))
            return self._create_response(b"", None)
        if getattr(chunk, "error", None):
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details(chunk.error)
            return self._create_response(b"", None)
        return self._chunk_to_response(chunk)
            
    async def SynthesizeStream(
        self, request: styletts_pb2.SynthesizeRequest, context: grpc.ServicerContext
    ) -> AsyncIterator[styletts_pb2.SynthesizeResponse]:
        model_id, voice_id, params = self._get_params(request)

        text = request.text

        stream = self.tts.create_stream(model_id, voice_id, params)
        stream.add_text(text)
        stream.force_generate()

        async for chunk in stream.pull_audio():
            if getattr(chunk, "error", None):
                context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
                context.set_details(chunk.error)
                break
            yield self._chunk_to_response(chunk)

        stream.close()

    async def SynthesizeIncremental(
        self,
        request_iterator: AsyncIterator[styletts_pb2.IncrementalSynthesizeRequest],
        context: grpc.ServicerContext,
    ) -> AsyncIterator[styletts_pb2.SynthesizeResponse]:
        config_received = False
        stream = None
        closed = False
        error_occurred = False

        async def request_handler():
            nonlocal config_received, stream, closed, error_occurred
            try:
                async for request in request_iterator:
                    if request.HasField("config"):
                        if config_received:
                            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
                            context.set_details("Config can only be sent once")
                            error_occurred = True
                            closed = True
                            return
                        config_received = True
                        model_id, voice_id, params = self._get_params(request)
                        stream = self.tts.create_stream(model_id, voice_id, params)
                    elif not config_received:
                        context.set_code(grpc.StatusCode.FAILED_PRECONDITION)
                        context.set_details("Config must be sent in first request")
                        error_occurred = True
                        closed = True
                        return
                    else:
                        if stream is None:
                            continue
                        if request.HasField("text_chunk"):
                            stream.add_text(request.text_chunk)
                        elif request.HasField("force_synthesis"):
                            stream.force_generate()
                        elif request.HasField("cancel"):
                            stream.cancel()
            except Exception as e:
                error_occurred = True
                closed = True
            finally:
                closed = True

        request_task = asyncio.create_task(request_handler())
        
        while not config_received and not closed:
            await asyncio.sleep(0.01)
        
        if error_occurred or stream is None:
            if not request_task.done():
                request_task.cancel()
            return
        
        try:
            while not closed or not request_task.done():
                has_audio = False
                async for chunk in stream.pull_audio():
                    has_audio = True
                    if getattr(chunk, "error", None):
                        context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
                        context.set_details(chunk.error)
                        error_occurred = True
                        break
                    response = self._chunk_to_response(chunk)
                    yield response
                
                if closed and request_task.done():
                    break
                
                if not has_audio:
                    await asyncio.sleep(0.01)
        finally:
            if stream is not None:
                stream.close()
            if not request_task.done():
                request_task.cancel()
