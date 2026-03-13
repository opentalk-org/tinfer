from __future__ import annotations
import asyncio
import json
import base64
import numpy as np
from typing import Any, Optional

from aiohttp.web_ws import WebSocketResponse
from tinfer.core.async_engine import AsyncStreamingTTS
from tinfer.core.request import AudioChunk, AlignmentType
from tinfer.utils.alignment_formatter import AlignmentFormatter
from tinfer.utils.audio_encoder import encode_audio_to_base64, parse_output_format, get_sample_rate


class WebSocketHandler:
    def __init__(
        self,
        ws: WebSocketResponse,
        tts: AsyncStreamingTTS,
        voice_id: str,
        query_params: dict[str, str],
    ) -> None:
        self.ws = ws
        self.tts = tts
        self.voice_id = voice_id
        self.query_params = query_params

        self.stream: Optional[Any] = None
        self.config_received = False
        self.closed = False
        self.alignment_formatter = AlignmentFormatter()

        self.model_id = query_params.get("model_id", "styletts2")
        self.output_format_str = query_params.get("output_format", "mp3_44100_32")
        self.output_format = parse_output_format(self.output_format_str)
        self.language_code = query_params.get("language_code")
        self.sync_alignment = query_params.get("sync_alignment", "false").lower() == "true"
        self.auto_mode = query_params.get("auto_mode", "false").lower() == "true"
        self.inactivity_timeout = int(query_params.get("inactivity_timeout", "20"))
        self.enable_logging = query_params.get("enable_logging", "true").lower() == "true"
        self.enable_ssml_parsing = query_params.get("enable_ssml_parsing", "false").lower() == "true"

        self.voice_settings: dict[str, Any] = {}
        self.chunk_length_schedule: list[int] = [120, 160, 250, 290]
        self.inactivity_task: Optional[asyncio.Task] = None
        self._text_added_event = asyncio.Event()
        self._stream_task: Optional[asyncio.Task] = None
        self._close_requested = False
        self._streaming_started = False

    async def handle(self) -> None:
        self._start_inactivity_timer()

        async for msg in self.ws:
            if self.closed or self.ws.closed:
                break
            if msg.type == 1:
                try:
                    data = json.loads(msg.data)
                    await self._process_message(data)
                except json.JSONDecodeError:
                    await self._send_error("Invalid JSON message")
                    break
                except Exception as e:
                    await self._send_error(f"Error processing message: {str(e)}")
                    break
            elif msg.type == 2:
                await self._send_error("Binary messages not supported")
                break
            elif msg.type == 257:
                break

    async def _process_message(self, data: dict[str, Any]) -> None:
        if not self.config_received:
            await self._handle_config_message(data)
        else:
            await self._handle_text_message(data)

    async def _handle_config_message(self, data: dict[str, Any]) -> None:
        voice_settings = data.get("voice_settings", {})
        generation_config = data.get("generation_config", {})

        self.voice_settings = voice_settings
        self.chunk_length_schedule = generation_config.get(
            "chunk_length_schedule",
            [120, 160, 250, 290],
        )

        params = self._map_params_to_tinfer(voice_settings, generation_config)

        try:
            self.stream = self.tts.create_stream(
                model_id=self.model_id,
                voice_id=self.voice_id,
                params=params,
            )
            self.config_received = True
            self._reset_inactivity_timer()
            self._stream_task = asyncio.create_task(self._stream_audio())
            text = data.get("text", "")
            if text and text.strip():
                self.stream.add_text(text)
                self.stream.force_generate()
                self._text_added_event.set()
        except Exception as e:
            await self._send_error(f"Failed to create stream: {str(e)}")
            self.closed = True

    async def _handle_text_message(self, data: dict[str, Any]) -> None:
        text = data.get("text", "")
        try_trigger_generation = data.get("try_trigger_generation", False)
        flush = data.get("flush", False)

        if not text.strip():
            if self.stream and try_trigger_generation:
                self.stream.force_generate()
                self._text_added_event.set()
            elif self.stream and not try_trigger_generation:
                self._close_requested = True
                self.stream.force_generate()
                self._text_added_event.set()
            self._reset_inactivity_timer()
            return

        if self.stream:
            self.stream.add_text(text)
            if try_trigger_generation or flush:
                self.stream.force_generate()
            self._text_added_event.set()
            self._reset_inactivity_timer()

    def _map_params_to_tinfer(
        self,
        voice_settings: dict[str, Any],
        generation_config: dict[str, Any],
    ) -> dict[str, Any]:
        params: dict[str, Any] = {}

        chunk_length_schedule = generation_config.get(
            "chunk_length_schedule",
            [120, 160, 250, 290],
        )

        params["chunk_length_schedule"] = chunk_length_schedule
        sample_rate = get_sample_rate(self.output_format)
        params["target_sample_rate"] = sample_rate
        params["target_encoding"] = self.output_format
        params["alignment_type"] = AlignmentType.CHAR if self.sync_alignment else AlignmentType.NONE
        return params

    async def _stream_audio(self) -> None:
        if not self.stream:
            return
        try:
            while not self.closed and not self.ws.closed:
                sent_any = False
                async for chunk in self.stream.pull_audio():
                    if self.closed or self.ws.closed:
                        break
                    if getattr(chunk, "error", None):
                        await self._send_error(chunk.error)
                        self.closed = True
                        break
                    response = await self._format_response(chunk)
                    await self._send_response(response)
                    sent_any = True
                if sent_any and not self.closed and not self.ws.closed:
                    should_exit = await self._send_final_if_needed()
                    if should_exit:
                        if self.stream:
                            try:
                                self.stream.close()
                            except Exception:
                                pass
                        break
                if self.closed or self.ws.closed:
                    break
                try:
                    await asyncio.wait_for(
                        self._text_added_event.wait(),
                        timeout=0.5,
                    )
                except asyncio.TimeoutError:
                    pass
                self._text_added_event.clear()
        except asyncio.CancelledError:
            pass
        except Exception as e:
            if not self.closed and not self.ws.closed:
                await self._send_error(f"Streaming error: {str(e)}")
            self.closed = True

    async def _format_response(self, chunk: AudioChunk) -> dict[str, Any]:
        if chunk.audio.dtype == np.uint8:
            audio_bytes = chunk.audio.tobytes()
            audio_base64 = base64.b64encode(audio_bytes).decode("utf-8")
        else:
            audio_base64 = encode_audio_to_base64(
                chunk.audio,
                chunk.sample_rate,
                self.output_format,
            )
        
        response: dict[str, Any] = {
            "audio": audio_base64,
        }
        if self.sync_alignment and chunk.alignments and chunk.alignments.items:
            char_data = self.alignment_formatter.to_websocket_format(
                chunk.alignments,
                normalized=True,
            )
            response["normalizedAlignment"] = char_data
            response["alignment"] = char_data
        return response

    async def _send_response(self, response: dict[str, Any]) -> None:
        if self.closed or self.ws.closed:
            return
        
        try:
            await self.ws.send_str(json.dumps(response))
            self._streaming_started = True
            self._reset_inactivity_timer()
        except Exception as e:
            if self.enable_logging:
                print(f"Error sending response: {e}")
            self.closed = True

    async def _send_error(self, error_message: str) -> None:
        if self.closed or self.ws.closed:
            return
        
        try:
            error_response = {
                "error": error_message,
            }
            await self.ws.send_str(json.dumps(error_response))
        except Exception:
            pass
        finally:
            self.closed = True

    async def _send_final_if_needed(self) -> bool:
        if self.closed or self.ws.closed:
            return False
        if not self._close_requested:
            return False
        try:
            await self.ws.send_str(json.dumps({"isFinal": True}))
        except Exception:
            pass
        return True

    def _start_inactivity_timer(self) -> None:
        self._reset_inactivity_timer()

    def _reset_inactivity_timer(self) -> None:
        if self.inactivity_task and not self.inactivity_task.done():
            self.inactivity_task.cancel()
        
        self.inactivity_task = asyncio.create_task(self._inactivity_timeout())

    async def _inactivity_timeout(self) -> None:
        try:
            await asyncio.sleep(self.inactivity_timeout)
            if self.closed or self.ws.closed:
                return
            if self._streaming_started:
                self._reset_inactivity_timer()
                return
            await self._send_error("Inactivity timeout")
            self.closed = True
        except asyncio.CancelledError:
            pass

    async def cleanup(self) -> None:
        self.closed = True
        self._text_added_event.set()

        if self.inactivity_task and not self.inactivity_task.done():
            self.inactivity_task.cancel()
            try:
                await self.inactivity_task
            except asyncio.CancelledError:
                pass

        if self._stream_task and not self._stream_task.done():
            self._stream_task.cancel()
            try:
                await self._stream_task
            except asyncio.CancelledError:
                pass

        if self.stream:
            try:
                self.stream.close()
            except Exception:
                pass
