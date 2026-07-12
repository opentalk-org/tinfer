import numpy as np

from tinfer.models.impl.styletts2.model.inference_config import StyleTTS2Params

from test_speed.benchmark.benchmark_corpus import (
    TokenCountingModel,
    configure_checkpoint_symbols,
    phonemize_training_text,
)
from test_speed.benchmark.benchmark_data import TextInput


def model_token_count(
    model: TokenCountingModel,
    text: str,
    language: str,
    use_training_phonemes: bool,
) -> int:
    model_text = (
        phonemize_training_text(text, language)
        if use_training_phonemes
        else text
    )
    return model._text_token_count(
        model_text,
        StyleTTS2Params(
            language=language,
            phonemized=use_training_phonemes,
        ),
    ) - 1


def build_balanced_grid(
    model: TokenCountingModel,
    passage: str,
    point_count: int,
    max_tokens: int,
    language: str,
    use_training_phonemes: bool,
) -> list[TextInput]:
    if use_training_phonemes:
        configure_checkpoint_symbols(
            model,
            language,
        )
    words = passage.split()
    assert words, "Balanced corpus requires passage words"
    minimum = min(
        model_token_count(model, word, language, use_training_phonemes)
        for word in words
    )
    targets = np.linspace(minimum, max_tokens, point_count)
    rows = []
    for index, target in enumerate(targets):
        offset = index * 37 % len(words)
        selected = []
        token_count = 0
        while True:
            candidate = [*selected, words[(offset + len(selected)) % len(words)]]
            candidate_text = " ".join(candidate)
            candidate_count = model_token_count(
                model,
                candidate_text,
                language,
                use_training_phonemes,
            )
            if candidate_count > target and selected:
                break
            assert candidate_count <= max_tokens
            selected = candidate
            token_count = candidate_count
            if candidate_count >= target:
                break
        rows.append(
            TextInput(
                text_id=f"balanced_{index:02d}_{token_count:03d}",
                text=" ".join(selected),
                input_phoneme_tokens=token_count,
            )
        )
    return rows
