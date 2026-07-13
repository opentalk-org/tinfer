from typing import Any

from tinfer.server.websocket.schemas import VoiceSettings


def map_styletts2_settings(
    settings: VoiceSettings,
    seed: int | None,
) -> dict[str, Any]:
    params: dict[str, Any] = {}
    if settings.speed is not None:
        params["speed"] = settings.speed
    if settings.stability is not None:
        params["style_interpolation_factor"] = settings.stability
    if settings.similarity_boost is not None:
        similarity_mix = 1.0 - settings.similarity_boost
        params["alpha"] = similarity_mix
        params["beta"] = similarity_mix
    if settings.style is not None:
        params["embedding_scale"] = 1.0 + settings.style
    if settings.alpha is not None:
        params["alpha"] = settings.alpha
    if settings.beta is not None:
        params["beta"] = settings.beta
    if seed is not None:
        params["seed"] = seed
    return params
