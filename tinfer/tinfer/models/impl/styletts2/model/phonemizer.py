from __future__ import annotations

import re
import threading
from typing import Any, Union

import espeak_align
from nltk.tokenize import TweetTokenizer
from tinfer.support.observability import get_logger

_PUNCTUATION = ';:,.!?¡¿—…"«»""'   
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
    ):
        self.language = language
        self.preserve_punctuation = preserve_punctuation
        self.with_stress = with_stress
        self.tie = tie
        self.espeak_workers = espeak_workers
        self._engine, self._engine_lock = _get_engine(language, tie, espeak_workers)
        self.t_tokenizer = TweetTokenizer()
        self._init_tokenizer()

    def _init_tokenizer(self) -> None:
        _pad = "$"
        _punctuation = ';:,.!?¡¿—…"«»"" '
        _letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
        _letters_ipa = "ɑɐɒæɓʙβɔɕçɗɖðʤəɘɚɛɜɝɞɟʄɡɠɢʛɦɧħɥʜɨɪʝɭɬɫɮʟɱɯɰŋɳɲɴøɵɸθœɶʘɹɺɾɻʀʁɽʂʃʈʧʉʊʋⱱʌɣɤʍχʎʏʑʐʒʔʡʕʢ\u0303\u032f\u032a\u0306ˈˌːˑʼʴʰʱʲʷˠˤ˞↓↑→↗↘'̩'\u0361"
        symbols = [_pad] + list(_punctuation) + list(_letters) + list(_letters_ipa)
        self.word_index_dictionary = {s: i for i, s in enumerate(symbols)}
        self.index_to_symbol = {i: s for s, i in self.word_index_dictionary.items()}

    def _preprocess_text(self, text: str) -> str:
        text = re.sub(re.compile(r'[„“”«»"]'), '"', text)
        text = re.sub(re.compile("[-—−‒‒–]"), "—", text)
        text = re.sub(re.compile(r"[\(\)\*\/\[\]]"), "", text)
        # text = re.sub(re.compile(r"[" + re.escape(';:,.!?¡¿—… \t\n""«»"" ') + r"]+$"), "", text).strip()
        return text

    def _preprocess_text_with_mapping(self, text: str) -> tuple[str, list[tuple[int, int]]]:
        chars: list[str] = []
        spans: list[tuple[int, int]] = []
        quote_chars = set('„“”«»"')
        dash_chars = set("-—−‒–")
        deleted_chars = set("()*/[]")

        for idx, char in enumerate(text):
            if char in deleted_chars:
                continue
            if char in quote_chars:
                chars.append('"')
            elif char in dash_chars:
                chars.append("—")
            else:
                chars.append(char)
            spans.append((idx, idx + 1))

        return "".join(chars), spans

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

        phonemes_list = [self._normalize_phoneme_string(p) for p in phonemes_list]
        phonemes_list = [self._filter_to_vocab(p) for p in phonemes_list]
        phonemized_string = self._normalize_phoneme_string("".join(phonemes_list))

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

        mapped: list[dict[str, Any]] = []
        consumed_original = 0

        for item in aligned:
            start = int(item["start"])
            end = int(item["end"])
            phonemes = self._filter_to_vocab(self._normalize_phoneme_string(str(item["phonemes"])))
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
        phonemized_string = self._normalize_phoneme_string(
            "".join(str(item["phonemes"]) for item in mapped)
        )
        return phonemized_string, mapped

    def tokenize(self, text: str) -> list[int]:
        return [self.word_index_dictionary[c] for c in text if c in self.word_index_dictionary]

    def detokenize(self, tokens: list[int]) -> str:
        return "".join(self.index_to_symbol[t] for t in tokens if t in self.index_to_symbol)

    def __call__(self, text: str) -> list[int]:
        return self.tokenize(text)
