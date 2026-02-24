from typing import Type

from tinfer.models.base.model import TTSModel

_MODEL_REGISTRY: dict[str, Type[TTSModel]] = {}


def register_model(model_id: str, model_class: Type[TTSModel]) -> None:
    _MODEL_REGISTRY[model_id] = model_class


def get_model_class(model_id: str) -> Type[TTSModel]:
    if model_id not in _MODEL_REGISTRY:
        raise ValueError(f"Model '{model_id}' is not registered. Available models: {list(_MODEL_REGISTRY.keys())}")
    return _MODEL_REGISTRY[model_id]

from tinfer.models.impl.styletts2.model.model import StyleTTS2

register_model("styletts2", StyleTTS2)

