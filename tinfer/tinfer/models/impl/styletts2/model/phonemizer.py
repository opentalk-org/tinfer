from __future__ import annotations

import re
import threading
from typing import Any, Sequence, Union

import espeak_align
from nltk.tokenize import TweetTokenizer
from tinfer.support.observability import get_logger

_PUNCTUATION = ';:,.!?¡¿—–…"«»""'
_ENGINE_CACHE: dict[tuple[str, bool, int], tuple[espeak_align.Engine, threading.Lock]] = {}
_engine_cache_lock = threading.Lock()
log = get_logger(__name__)


def _get_engine(language: str, tie: bool, espeak_workers: int) -> tuple[espeak_align.Engine, threading.Lock]:
    key = (language, tie, espeak_workers)
    with _engine_cache_lock:
        if key not in _ENGINE_CACHE:
            eng = espeak_align.Engine(language, tie=tie, espeak_workers=espeak_workers)
            _ENGINE_CACHE[key] = (eng, threading.Lock())
        return _ENGINE_CACHE[key]


class StyleTTS2Phonemizer:
    def __init__(
        self,
        language: str = "pl",
        preserve_punctuation: bool = True,
        with_stress: bool = True,
        tie: bool = True,
        espeak_workers: int = 4,
        symbols: Sequence[str] | None = None,
    ):
        self.language = language
        self.preserve_punctuation = preserve_punctuation
        self.with_stress = with_stress
        self.tie = tie
        self.espeak_workers = espeak_workers
        self._engine, self._engine_lock = _get_engine(language, tie, espeak_workers)
        self.t_tokenizer = TweetTokenizer()
        self._init_tokenizer(symbols)

    def _init_tokenizer(self, symbols: Sequence[str] | None) -> None:
        _pad = "$"
        _punctuation = ';:,.!?¡¿—…"«»“” '
        _letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
        _letters_ipa = "ɑɐɒæɓʙβɔɕçɗɖðʤəɘɚɛɜɝɞɟʄɡɠɢʛɦɧħɥʜɨɪʝɭɬɫɮʟɱɯɰŋɳɲɴøɵɸθœɶʘɹɺɾɻʀʁɽʂʃʈʧʉʊʋⱱʌɣɤʍχʎʏʑʐʒʔʡʕʢ\u0303\u032f\u032a\u0306ˈˌːˑʼʴʰʱʲʷˠˤ˞↓↑→↗↘'̩'\u0361"
        if symbols is None:
            symbols = [_pad] + list(_punctuation) + list(_letters) + list(_letters_ipa)
        self.word_index_dictionary = {s: i for i, s in enumerate(symbols)}
        self.index_to_symbol = {i: s for s, i in self.word_index_dictionary.items()}

    def _preprocess_text(self, text: str) -> str:
        text = re.sub(r"\s\s*", " ", text)
        if not re.search(r"[\.\?!]$", text):
            text = f"{text}."
        text = re.sub(re.compile(r'[„“”«»"]'), '"', text)
        text = re.sub(re.compile("[-—−‒–]"), "—", text)
        text = re.sub(re.compile(r"[\(\)\*\/\[\]]"), "", text)
        return " ".join(self.t_tokenizer.tokenize(text))

    def _preprocess_text_with_mapping(self, text: str) -> tuple[str, list[tuple[int, int]]]:
        chars: list[str] = []
        spans: list[tuple[int, int]] = []
        quote_chars = set('„“”«»"')
        dash_chars = set("-—−‒–")
        deleted_chars = set("()*/[]")

        idx = 0
        while idx < len(text):
            char = text[idx]
            if char.isspace():
                start = idx
                while idx < len(text) and text[idx].isspace():
                    idx += 1
                chars.append(" ")
                spans.append((start, idx))
                continue
            chars.append(char)
            spans.append((idx, idx + 1))
            idx += 1

        if not re.search(r"[\.\?!]$", "".join(chars)):
            chars.append(".")
            spans.append((len(text), len(text)))

        normalized_chars: list[str] = []
        normalized_spans: list[tuple[int, int]] = []
        for char, span in zip(chars, spans):
            if char in deleted_chars:
                continue
            if char in quote_chars:
                normalized_chars.append('"')
            elif char in dash_chars:
                normalized_chars.append("—")
            else:
                normalized_chars.append(char)
            normalized_spans.append(span)

        normalized = "".join(normalized_chars)
        token_spans: list[tuple[int, int]] = []
        cursor = 0
        for token in self.t_tokenizer.tokenize(normalized):
            start = normalized.find(token, cursor)
            if start < 0:
                raise ValueError(f"TweetTokenizer token not found in preprocessed text: {token}")
            end = start + len(token)
            token_spans.append((start, end))
            cursor = end
        tokenized_chars: list[str] = []
        tokenized_spans: list[tuple[int, int]] = []
        previous_end = 0

        for start, end in token_spans:
            if tokenized_chars:
                current_start = normalized_spans[start][0]
                previous_original_end = normalized_spans[previous_end - 1][1]
                tokenized_chars.append(" ")
                tokenized_spans.append((previous_original_end, max(previous_original_end, current_start)))
            tokenized_chars.extend(normalized_chars[start:end])
            tokenized_spans.extend(normalized_spans[start:end])
            previous_end = end

        return "".join(tokenized_chars), tokenized_spans

    @staticmethod
    def _normalize_phoneme_string(s: str) -> str:
        s = s.replace("``", '"').replace("''", '"')
        return " ".join(s.split())

    def _filter_to_vocab(self, s: str) -> str:
        return "".join(c for c in s if c in self.word_index_dictionary)

    def process_text(
        self,
        text: str,
        phonemize: bool = True,
        word_alignment: bool = False,
    ) -> Union[str, tuple[str, tuple[list[str], list[str]]]]:
        text = text.strip()
        if not phonemize:
            return text

        preprocessed = self._preprocess_text(text)
        with self._engine_lock:
            words, phonemes_list = self._engine.align(preprocessed, _PUNCTUATION, threads=8)

        log.debug(
            "rust_phonemizer_output",
            text=text,
            preprocessed=preprocessed,
            words=words,
            phonemes=phonemes_list,
        )

        phonemes_list = [self._normalize_phoneme_string(p) for p in phonemes_list]
        phonemized_string = self._normalize_phoneme_string(
            " ".join(phoneme for phoneme in phonemes_list if phoneme)
        )

        log.debug(
            "text_phonemized",
            phonemized=phonemized_string,
            words=words,
            phonemes=phonemes_list,
        )

        if word_alignment:
            return phonemized_string, (words, phonemes_list)
        return phonemized_string

    def align_text_with_original_spans(self, text: str) -> list[dict[str, Any]]:
        preprocessed, preprocessed_to_original = self._preprocess_text_with_mapping(text)
        with self._engine_lock:
            aligned = self._engine.align_with_spans(preprocessed, _PUNCTUATION, threads=8)

        log.debug(
            "rust_phonemizer_output",
            text=text,
            preprocessed=preprocessed,
            items=[dict(item) for item in aligned],
        )

        mapped: list[dict[str, Any]] = []
        consumed_original = 0

        for item in aligned:
            start = int(item["start"])
            end = int(item["end"])
            phonemes = self._normalize_phoneme_string(str(item["phonemes"]))
            if start < end:
                original_start = preprocessed_to_original[start][0]
                original_end = preprocessed_to_original[end - 1][1]
            else:
                original_start = consumed_original
                original_end = consumed_original

            if consumed_original < original_start:
                mapped.append(
                    {
                        "token": "",
                        "phonemes": "",
                        "preprocessed_start": start,
                        "preprocessed_end": start,
                        "original_start": consumed_original,
                        "original_end": original_start,
                        "original_text": text[consumed_original:original_start],
                    }
                )

            mapped.append(
                {
                    "token": item["token"],
                    "phonemes": phonemes,
                    "preprocessed_start": start,
                    "preprocessed_end": end,
                    "original_start": original_start,
                    "original_end": original_end,
                    "original_text": text[original_start:original_end],
                }
            )
            consumed_original = max(consumed_original, original_end)

        if consumed_original < len(text):
            mapped.append(
                {
                    "token": "",
                    "phonemes": "",
                    "preprocessed_start": len(preprocessed_to_original),
                    "preprocessed_end": len(preprocessed_to_original),
                    "original_start": consumed_original,
                    "original_end": len(text),
                    "original_text": text[consumed_original:],
                }
            )

        return mapped

    def process_text_with_original_spans(self, text: str) -> tuple[str, list[dict[str, Any]]]:
        mapped = self.align_text_with_original_spans(text)
        phoneme_items = [str(item["phonemes"]) for item in mapped if item["phonemes"]]
        phonemized_string = self._normalize_phoneme_string(
            " ".join(phoneme_items)
        )
        mapped_for_alignment: list[dict[str, Any]] = []
        non_empty_seen = 0
        last_non_empty = len(phoneme_items) - 1
        for item in mapped:
            mapped_item = dict(item)
            if mapped_item["phonemes"]:
                if non_empty_seen < last_non_empty:
                    mapped_item["phonemes"] = f"{mapped_item['phonemes']} "
                non_empty_seen += 1
            mapped_item["phonemes"] = self._filter_to_vocab(str(mapped_item["phonemes"]))
            mapped_for_alignment.append(mapped_item)
        log.debug(
            "text_phonemized",
            text=text,
            phonemized=phonemized_string,
            words=[item["original_text"] for item in mapped_for_alignment],
            phonemes=[item["phonemes"] for item in mapped_for_alignment],
            mapping=mapped_for_alignment,
        )
        return phonemized_string, mapped_for_alignment

    def tokenize(self, text: str) -> list[int]:
        return [self.word_index_dictionary[c] for c in text if c in self.word_index_dictionary]

    def detokenize(self, tokens: list[int]) -> str:
        return "".join(self.index_to_symbol[t] for t in tokens if t in self.index_to_symbol)

    def __call__(self, text: str) -> list[int]:
        return self.tokenize(text)
