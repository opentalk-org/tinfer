import argparse
import json
from pathlib import Path

from tinfer.models.impl.styletts2.model.text_config import TextConfig


def add_text_config_arguments(parser: argparse.ArgumentParser, required: bool) -> None:
    parser.add_argument("--symbols-file", required=required)
    parser.add_argument("--supported-languages", nargs="+", required=required)
    parser.add_argument("--default-language", required=required)


def text_config_from_args(
    args: argparse.Namespace,
    expected_symbol_count: int,
) -> TextConfig | None:
    values = (args.symbols_file, args.supported_languages, args.default_language)
    if not any(values):
        return None
    if not all(values):
        raise ValueError("symbols, supported languages, and default language must be provided together")

    return load_text_config(
        args.symbols_file,
        args.supported_languages,
        args.default_language,
        expected_symbol_count,
    )


def load_text_config(
    symbols_path: str,
    supported_languages: list[str],
    default_language: str,
    expected_symbol_count: int,
) -> TextConfig:
    with Path(symbols_path).open(encoding="utf-8") as symbols_file:
        symbols = json.load(symbols_file)
    return TextConfig.from_dict(
        {
            "symbols": symbols,
            "supported_languages": supported_languages,
            "default_language": default_language,
        },
        expected_symbol_count,
    )
