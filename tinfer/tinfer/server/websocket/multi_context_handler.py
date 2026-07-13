from __future__ import annotations

import asyncio
from dataclasses import replace
from functools import partial
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
from tinfer.server.websocket.schemas import Transport
from tinfer.server.websocket.speech_parser import parse_speech_request
from tinfer.server.websocket.stream_context import StreamContext

CLIENT_MESSAGE_FIELDS = {
    "text",
    "context_id",
    "voice_settings",
    "generation_config",
    "pronunciation_dictionary_locators",
    "xi_api_key",
    "authorization",
    "flush",
    "close_context",
    "close_socket",
}


class MultiContextHandler:
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
        query = parse_query(_QueryRequest(query_params), Transport.WEBSOCKET)
        self.model_info = models.resolve(query.model_id)
        self.query = replace(query, model_id=self.model_info.model_id)
        if voice_id not in self.tts.get_voice_ids(self.query.model_id):
            raise ValueError(f"unknown voice: {voice_id}")
        self.language = validate_language(self.query, self.model_info)
        self.contexts: dict[str, StreamContext] = {}
        self._initialized = False
        self._closing = False
        self._outbound: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue()
        self._writer_task = asyncio.create_task(self._write_responses())

    async def handle(self) -> None:
        try:
            async for message in self.ws:
                if message.type == WSMsgType.TEXT:
                    await self._dispatch_json(message.data)
                    if self._closing or self.ws.closed:
                        break
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
            await self._enqueue({"error": str(error)})
            await self._outbound.join()
            await self.ws.close(code=1008)

    async def cleanup(self) -> None:
        contexts = list(self.contexts.values())
        self.contexts.clear()
        try:
            await asyncio.gather(*(context.close() for context in contexts))
        finally:
            await self._outbound.put(None)
            await self._writer_task

    def abort(self) -> None:
        contexts = list(self.contexts.values())
        self.contexts.clear()
        for context in contexts:
            context.abort()
        if not self._writer_task.done():
            self._writer_task.cancel()

    async def _dispatch_json(self, raw_message: str) -> None:
        payload = json.loads(raw_message)
        if not isinstance(payload, dict):
            raise ValueError("message must be a JSON object")
        unknown_fields = set(payload) - CLIENT_MESSAGE_FIELDS
        if unknown_fields:
            raise ValueError(f"unsupported message field: {sorted(unknown_fields)[0]}")
        if not self._initialized:
            if "text" not in payload or payload["text"] != " ":
                raise ValueError('first message text must be exactly " "')
            context_id = payload["context_id"] if "context_id" in payload else "default"
            self._validate_context_id(context_id)
            await self._create_context(context_id, payload)
            self._initialized = True
            return
        close_socket = payload["close_socket"] if "close_socket" in payload else False
        if not isinstance(close_socket, bool):
            raise ValueError("close_socket must be boolean")
        if close_socket:
            await self._finalize_socket()
            return
        context_id = payload["context_id"] if "context_id" in payload else "default"
        self._validate_context_id(context_id)
        if context_id not in self.contexts:
            if "text" not in payload:
                raise ValueError(f"unknown context: {context_id}")
            close_context = payload["close_context"] if "close_context" in payload else False
            if not isinstance(close_context, bool):
                raise ValueError("close_context must be boolean")
            if close_context:
                raise ValueError(f"unknown context: {context_id}")
            await self._create_context(context_id, payload)
            return
        await self._update_context(context_id, payload)

    def _validate_context_id(self, context_id: Any) -> None:
        if not isinstance(context_id, str):
            raise ValueError("context_id must be a string")
        if not context_id:
            raise ValueError("context_id must not be empty")

    async def _create_context(self, context_id: str, payload: dict[str, Any]) -> None:
        speech_payload: dict[str, Any] = {"text": payload["text"]}
        if "voice_settings" in payload:
            speech_payload["voice_settings"] = payload["voice_settings"]
        if "generation_config" in payload:
            speech_payload["generation_config"] = payload["generation_config"]
        speech = parse_speech_request(speech_payload)
        if speech.text not in ("", " ") and not speech.text.endswith(" "):
            raise ValueError("text must end in a space")
        alignment_type = AlignmentType.CHAR if self.query.sync_alignment else AlignmentType.NONE
        params = map_stream_params(self.query, speech, alignment_type)
        params["tts_params"]["language"] = self.language
        stream = self.tts.create_stream(self.query.model_id, self.voice_id, params)
        context = StreamContext(
            stream=stream,
            output_format=self.query.output_format,
            send_response=self._enqueue,
            inactivity_timeout=self.query.inactivity_timeout,
            close_response=partial(self._expire_context, context_id),
            context_id=context_id,
        )
        self.contexts[context_id] = context
        if speech.text not in ("", " "):
            flush = payload["flush"] if "flush" in payload else False
            if not isinstance(flush, bool):
                raise ValueError("flush must be boolean")
            await context.add_text(speech.text)
            if flush:
                await context.flush()

    async def _update_context(self, context_id: str, payload: dict[str, Any]) -> None:
        close_context = payload["close_context"] if "close_context" in payload else False
        if not isinstance(close_context, bool):
            raise ValueError("close_context must be boolean")
        context = self.contexts[context_id]
        if close_context:
            await context.finalize()
            del self.contexts[context_id]
            return
        if "voice_settings" in payload or "generation_config" in payload:
            await context.finalize()
            del self.contexts[context_id]
            reinitialized = dict(payload)
            if "text" not in reinitialized:
                reinitialized["text"] = " "
            await self._create_context(context_id, reinitialized)
            return
        flush = payload["flush"] if "flush" in payload else False
        if not isinstance(flush, bool):
            raise ValueError("flush must be boolean")
        if "text" not in payload and flush:
            await context.flush()
            return
        if "text" not in payload or not isinstance(payload["text"], str):
            raise ValueError("text is required and must be a string")
        text = payload["text"]
        if text == "":
            await context.keepalive()
            return
        if not text.endswith(" "):
            raise ValueError("text must end in a space")
        if text == " ":
            if flush:
                await context.flush()
            else:
                await context.keepalive()
            return
        await context.add_text(text)
        if flush:
            await context.flush()

    async def _finalize_socket(self) -> None:
        self._closing = True
        contexts = list(self.contexts.values())
        self.contexts.clear()
        await asyncio.gather(*(context.finalize() for context in contexts))
        await self._outbound.join()

    async def _expire_context(self, context_id: str) -> None:
        if context_id in self.contexts:
            context = self.contexts.pop(context_id)
            await context.close()

    async def _enqueue(self, payload: dict[str, Any]) -> None:
        await self._outbound.put(payload)

    async def _write_responses(self) -> None:
        while True:
            payload = await self._outbound.get()
            try:
                if payload is None:
                    return
                if not self.ws.closed:
                    await self.ws.send_str(json.dumps(payload, separators=(",", ":")))
            finally:
                self._outbound.task_done()


class _QueryRequest:
    def __init__(self, query: dict[str, str]) -> None:
        self.query = query
