from tinfer.core.request import ModelInfo


class ModelResolver:
    def __init__(self, model_infos: list[ModelInfo], default_model_id: str) -> None:
        self._models = {info.model_id: info for info in model_infos}
        if default_model_id not in self._models:
            raise ValueError(f"unknown default model: {default_model_id}")
        self._default_model_id = default_model_id

    def resolve(self, requested_id: str | None) -> ModelInfo:
        model_id = requested_id if requested_id is not None else self._default_model_id
        if model_id not in self._models:
            raise ValueError(f"unknown model: {model_id}")
        return self._models[model_id]
