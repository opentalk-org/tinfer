from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class TextConfig:
    symbols: tuple[str, ...]
    supported_languages: tuple[str, ...]
    default_language: str

    @classmethod
    def from_dict(
        cls,
        value: dict[str, Any],
        expected_symbol_count: int,
    ) -> "TextConfig":
        symbols = tuple(value["symbols"])
        supported_languages = tuple(value["supported_languages"])
        default_language = value["default_language"]

        if len(symbols) != expected_symbol_count:
            raise ValueError(
                f"text config has {len(symbols)} symbols, model expects {expected_symbol_count}"
            )
        if not symbols or any(not isinstance(symbol, str) or not symbol for symbol in symbols):
            raise ValueError("text config symbols must be non-empty strings")
        if not supported_languages or len(set(supported_languages)) != len(supported_languages):
            raise ValueError("supported languages must be a non-empty unique list")
        if default_language not in supported_languages:
            raise ValueError("default language must be included in supported languages")

        return cls(symbols, supported_languages, default_language)

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbols": list(self.symbols),
            "supported_languages": list(self.supported_languages),
            "default_language": self.default_language,
        }
