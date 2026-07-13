from __future__ import annotations

import grpc

from tinfer.support.observability import get_logger
from . import styletts_pb2

log = get_logger(__name__)


class CatalogServiceMixin:
    async def ListModels(self, request: styletts_pb2.ListModelsRequest, context: grpc.ServicerContext) -> styletts_pb2.ListModelsResponse:
        model_infos = self.tts.get_model_infos()
        response = styletts_pb2.ListModelsResponse()
        for info in model_infos:
            model = response.models.add()
            model.model_id = info.model_id
            model.supported_languages.extend(info.supported_languages)
            model.default_language = info.default_language
        log.info("grpc_list_models", model_count=len(response.models))
        return response

    async def ListVoices(self, request: styletts_pb2.ListVoicesRequest, context: grpc.ServicerContext) -> styletts_pb2.ListVoicesResponse:
        model_ids = [request.model_id] if request.model_id else self.tts.get_model_ids()
        response = styletts_pb2.ListVoicesResponse()
        for model_id in model_ids:
            try:
                voice_ids = self.tts.get_voice_ids(model_id)
            except ValueError as e:
                context.set_code(grpc.StatusCode.NOT_FOUND)
                context.set_details(str(e))
                return response
            for voice_id in voice_ids:
                voice = response.voices.add()
                voice.model_id = model_id
                voice.voice_id = voice_id
        log.info("grpc_list_voices", model_count=len(model_ids), voice_count=len(response.voices))
        return response

