from __future__ import annotations

import asyncio
from dataclasses import replace
import json

from aiohttp import web
import numpy as np

from tinfer.core.request import Alignment, AlignmentType, AudioChunk
from tinfer.server.health import HealthState
from tinfer.server.websocket.errors import RequestValidationError, validation_response
from tinfer.server.websocket.model_resolver import ModelResolver
from tinfer.server.websocket.query_parser import (
    map_stream_params,
    parse_query,
    validate_language,
)
from tinfer.server.websocket.response_formatter import (
    encode_chunk,
    format_http_timing,
)
from tinfer.server.websocket.schemas import SpeechOutputFormat, Transport
from tinfer.server.websocket.speech_parser import parse_speech_request


class SpeechHttpHandler:
    def __init__(self, tts, health: HealthState, models: ModelResolver) -> None:
        self.tts = tts
        self.health = health
        self.models = models

    async def unary_audio(self, request: web.Request) -> web.Response:
        return await self._unary(request, with_timing=False)

    async def unary_timing(self, request: web.Request) -> web.Response:
        return await self._unary(request, with_timing=True)

    async def stream_audio(self, request: web.Request) -> web.StreamResponse:
        return await self._stream(request, with_timing=False)

    async def stream_timing(self, request: web.Request) -> web.StreamResponse:
        return await self._stream(request, with_timing=True)

    async def _unary(self, request: web.Request, with_timing: bool) -> web.Response:
        if not await self.health.try_acquire_connection():
            return self._error(503, "server is not accepting synthesis requests")
        stream = None
        try:
            query, speech, model_info = await self._prepare(request)
            alignment_type = AlignmentType.CHAR if with_timing else AlignmentType.NONE
            params = map_stream_params(query, speech, alignment_type)
            params["tts_params"]["language"] = validate_language(query, model_info)
            stream = self.tts.create_stream(query.model_id, request.match_info["voice_id"], params)
            stream.add_text(speech.text)
            stream.force_generate()
            chunks = [chunk async for chunk in stream.pull_audio()]
            merged = self._merge_chunks(chunks)
            if with_timing:
                return web.json_response(format_http_timing(merged, query.output_format))
            return web.Response(
                body=encode_chunk(merged, query.output_format),
                content_type=(
                    "audio/wav"
                    if query.output_format.value.startswith("wav_")
                    else "application/octet-stream"
                ),
            )
        except web.HTTPException as error:
            return self._error(error.status, error.reason)
        except RequestValidationError as error:
            return validation_response(error)
        except (json.JSONDecodeError, TypeError, ValueError) as error:
            return self._error(422, str(error))
        finally:
            if stream is not None:
                stream.close()
            await self.health.release_connection()

    async def _stream(self, request: web.Request, with_timing: bool) -> web.StreamResponse:
        if not await self.health.try_acquire_connection():
            return self._error(503, "server is not accepting synthesis requests")
        stream = None
        response = None
        try:
            query, speech, model_info = await self._prepare(request)
            if query.output_format.value.startswith("wav_"):
                raise ValueError("WAV output is only available for non-streaming requests")
            alignment_type = AlignmentType.CHAR if with_timing else AlignmentType.NONE
            params = map_stream_params(query, speech, alignment_type)
            params["tts_params"]["language"] = validate_language(query, model_info)
            stream = self.tts.create_stream(query.model_id, request.match_info["voice_id"], params)
            response = web.StreamResponse(
                status=200,
                headers={"Content-Type": "text/event-stream" if with_timing else _audio_mime(query.output_format)},
            )
            await response.prepare(request)
            stream.add_text(speech.text)
            stream.force_generate()
            async for chunk in stream.pull_audio():
                if with_timing:
                    payload = json.dumps(
                        format_http_timing(chunk, query.output_format),
                        separators=(",", ":"),
                    ).encode("utf-8") + b"\n"
                else:
                    payload = encode_chunk(chunk, query.output_format)
                await response.write(payload)
            await response.write_eof()
            return response
        except web.HTTPException as error:
            if response is None:
                return self._error(error.status, error.reason)
            response.force_close()
            return response
        except RequestValidationError as error:
            if response is None:
                return validation_response(error)
            response.force_close()
            return response
        except (json.JSONDecodeError, TypeError, ValueError) as error:
            if response is None:
                return self._error(422, str(error))
            response.force_close()
            return response
        except (ConnectionResetError, asyncio.CancelledError):
            raise
        finally:
            if stream is not None:
                stream.close()
            await self.health.release_connection()

    async def _prepare(self, request: web.Request):
        try:
            payload = await request.json(loads=json.loads)
        except json.JSONDecodeError as error:
            raise web.HTTPUnprocessableEntity(reason="request body must be JSON") from error
        if not isinstance(payload, dict):
            raise web.HTTPUnprocessableEntity(reason="request body must be an object")
        speech = parse_speech_request(payload)
        if not speech.text.strip():
            raise web.HTTPUnprocessableEntity(reason="text must contain speech content")
        query = parse_query(request, Transport.HTTP)
        if speech.model_id is not None:
            query = replace(query, model_id=speech.model_id)
        if speech.language_code is not None:
            query = replace(query, language_code=speech.language_code)
        try:
            model_info = self.models.resolve(query.model_id)
        except ValueError as error:
            raise web.HTTPNotFound(reason=str(error)) from error
        query = replace(query, model_id=model_info.model_id)
        voice_id = request.match_info["voice_id"]
        if voice_id not in self.tts.get_voice_ids(query.model_id):
            raise web.HTTPNotFound(reason=f"unknown voice: {voice_id}")
        validate_language(query, model_info)
        return query, speech, model_info

    def _merge_chunks(self, chunks: list[AudioChunk]) -> AudioChunk:
        if not chunks:
            raise ValueError("synthesis returned no audio")
        failed = next((chunk.error for chunk in chunks if chunk.error is not None), None)
        if failed is not None:
            raise RuntimeError(failed)
        sample_rate = chunks[0].sample_rate
        if any(chunk.sample_rate != sample_rate for chunk in chunks):
            raise RuntimeError("synthesis chunks have inconsistent sample rates")
        alignments = [
            item
            for chunk in chunks
            if chunk.alignments is not None
            for item in chunk.alignments.items
        ]
        return AudioChunk(
            audio=np.concatenate([chunk.audio for chunk in chunks]),
            sample_rate=sample_rate,
            alignments=Alignment(alignments, AlignmentType.CHAR),
        )

    def _error(self, status: int, message: str) -> web.Response:
        return web.json_response({"detail": {"status": status, "message": message}}, status=status)


def _audio_mime(output_format: SpeechOutputFormat) -> str:
    if output_format.value.startswith("mp3"):
        return "audio/mpeg"
    if output_format.value.startswith("opus"):
        return "audio/ogg"
    if output_format.value.startswith("pcm"):
        return "audio/pcm"
    if output_format.value.startswith("wav"):
        return "audio/wav"
    return "application/octet-stream"
