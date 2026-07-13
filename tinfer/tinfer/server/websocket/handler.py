from __future__ import annotations

import asyncio
from dataclasses import replace
import json
from typing import Any

from aiohttp import WSMsgType
from aiohttp.web_ws import WebSocketResponse

from tinfer.core.request import AlignmentType
from tinfer.server.websocket.model_resolver import ModelResolver
from tinfer.server.websocket.query_parser import (
    map_stream_params,
    parse_query,
    validate_language,
)
from tinfer.server.websocket.speech_parser import parse_speech_request
from tinfer.server.websocket.stream_context import StreamContext
from tinfer.server.websocket.schemas import GenerationConfig, Transport, VoiceSettings

INITIAL_IGNORED_FIELDS = {
    "xi-api-key",
    "authorization",
    "pronunciation_dictionary_locators",
}
CLIENT_MESSAGE_FIELDS = {
    "text",
    "try_trigger_generation",
    "voice_settings",
    "generator_config",
    "flush",
}


class WebSocketHandler:
    def __init__(
        self,
        ws: WebSocketResponse,
        tts,
        voice_id: str,
        query_params: dict[str, str],
        models: ModelResolver,
    ) -> None:
        self.ws = ws
        self.tts = tts
        self.voice_id = voice_id
        self.query_params = query_params
        self.models = models
        self.context: StreamContext | None = None
        self._send_lock = asyncio.Lock()
        self._initialized = False
        self._voice_settings: VoiceSettings | None = None
        self._generation_config: GenerationConfig | None = None

    async def handle(self) -> None:
        try:
            async for message in self.ws:
                if message.type == WSMsgType.TEXT:
                    await self._dispatch_json(message.data)
                elif message.type == WSMsgType.BINARY:
                    raise ValueError("binary messages are not supported")
                elif message.type in (
                    WSMsgType.CLOSE,
                    WSMsgType.CLOSING,
                    WSMsgType.CLOSED,
                    WSMsgType.ERROR,
                ):
                    break
        except (json.JSONDecodeError, TypeError, ValueError) as error:
            await self._send({"error": str(error)})
            await self.ws.close(code=1008)

    async def cleanup(self) -> None:
        if self.context is not None:
            await self.context.close()

    def abort(self) -> None:
        if self.context is not None:
            self.context.abort()

    async def _dispatch_json(self, raw_message: str) -> None:
        payload = json.loads(raw_message)
        if not isinstance(payload, dict):
            raise ValueError("message must be a JSON object")
        if not self._initialized:
            await self._initialize(payload)
            return
        await self._dispatch_text(payload)

    async def _initialize(self, payload: dict[str, Any]) -> None:
        if "text" not in payload or payload["text"] != " ":
            raise ValueError('first message text must be exactly " "')
        query = parse_query(_QueryRequest(self.query_params), Transport.WEBSOCKET)
        model_info = self.models.resolve(query.model_id)
        query = replace(query, model_id=model_info.model_id)
        if self.voice_id not in self.tts.get_voice_ids(query.model_id):
            raise ValueError(f"unknown voice: {self.voice_id}")
        language = validate_language(query, model_info)
        speech_payload = dict(payload)
        for field in INITIAL_IGNORED_FIELDS:
            speech_payload.pop(field, None)
        speech = parse_speech_request(speech_payload)
        self._voice_settings = speech.voice_settings
        self._generation_config = speech.generation_config
        alignment_type = AlignmentType.CHAR if query.sync_alignment else AlignmentType.NONE
        params = map_stream_params(query, speech, alignment_type)
        params["tts_params"]["language"] = language
        stream = self.tts.create_stream(query.model_id, self.voice_id, params)
        self.context = StreamContext(
            stream=stream,
            output_format=query.output_format,
            send_response=self._send,
            inactivity_timeout=query.inactivity_timeout,
            close_response=self._close_socket,
        )
        self._initialized = True

    async def _dispatch_text(self, payload: dict[str, Any]) -> None:
        assert self.context is not None, "initialized handler requires a stream context"
        unknown_fields = set(payload) - CLIENT_MESSAGE_FIELDS
        if unknown_fields:
            raise ValueError(f"unsupported message field: {sorted(unknown_fields)[0]}")
        if "text" not in payload or not isinstance(payload["text"], str):
            raise ValueError("text is required and must be a string")
        text = payload["text"]
        self._validate_settings(payload)
        flush = payload["flush"] if "flush" in payload else False
        trigger = (
            payload["try_trigger_generation"]
            if "try_trigger_generation" in payload
            else False
        )
        if not isinstance(flush, bool):
            raise ValueError("flush must be boolean")
        if not isinstance(trigger, bool):
            raise ValueError("try_trigger_generation must be boolean")
        if text == "":
            await self.context.finalize()
            await self.ws.close()
            return
        if not text.endswith(" "):
            raise ValueError("text must end in a space")
        if text == " ":
            if flush:
                await self.context.flush()
            elif trigger:
                await self.context.try_generate()
            else:
                await self.context.keepalive()
            return
        await self.context.add_text(text)
        if flush:
            await self.context.flush()
        elif trigger:
            await self.context.try_generate()

    def _validate_settings(self, payload: dict[str, Any]) -> None:
        if "generation_config" in payload:
            raise ValueError("generation_config is not a client message field")
        if "voice_settings" in payload:
            parsed = parse_speech_request(
                {"text": payload["text"], "voice_settings": payload["voice_settings"]}
            )
            if parsed.voice_settings != self._voice_settings:
                raise ValueError("voice_settings cannot change after initialization")
        if "generator_config" in payload:
            parsed = parse_speech_request(
                {"text": payload["text"], "generation_config": payload["generator_config"]}
            )
            if parsed.generation_config != self._generation_config:
                raise ValueError("generator_config cannot change after initialization")

    async def _send(self, payload: dict[str, Any]) -> None:
        async with self._send_lock:
            if not self.ws.closed:
                await self.ws.send_str(json.dumps(payload, separators=(",", ":")))

    async def _close_socket(self) -> None:
        await self.ws.close(code=1000)


class _QueryRequest:
    def __init__(self, query: dict[str, str]) -> None:
        self.query = query
