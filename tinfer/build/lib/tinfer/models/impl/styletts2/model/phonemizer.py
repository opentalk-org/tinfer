from __future__ import annotations

import re
import threading
from typing import Union

import espeak_align
from nltk.tokenize import TweetTokenizer

_PUNCTUATION = ';:,.!?¡¿—…"«»""'   
_ENGINE_CACHE: dict[tuple[str, bool, int], tuple[espeak_align.Engine, threading.Lock]] = {}
_engine_cache_lock = threading.Lock()


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
        espeak_workers: int = 4, # 80 Mb RAM total
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
        _letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
        _letters_ipa = "ɑɐɒæɓʙβɔɕçɗɖðʤəɘɚɛɜɝɞɟʄɡɠɢʛɦɧħɥʜɨɪʝɭɬɫɮʟɱɯɰŋɳɲɴøɵɸθœɶʘɹɺɾɻʀʁɽʂʃʈʧʉʊʋⱱʌɣɤʍχʎʏʑʐʒʔʡʕʢ\u0303\u032f\u032a\u0306ˈˌːˑʼʴʰʱʲʷˠˤ˞↓↑→↗↘'̩'\u0361"
        symbols = [_pad] + list(_punctuation) + list(_letters) + list(_letters_ipa)
        self.word_index_dictionary = {symbol: i for i, symbol in enumerate(symbols)}
        self.index_to_symbol = {i: symbol for symbol, i in self.word_index_dictionary.items()}

    def _preprocess_text(self, text: str) -> str:
        text = re.sub(re.compile(r'[„""""«»"]'), '"', text)
        text = re.sub(re.compile('[-—−‒‒–]'), '—', text)
        text = re.sub(re.compile(r'[\(\)\*\/\[\]]'), '', text)
        # text = re.sub(re.compile(r"[" + re.escape(';:,.!?¡¿—… \t\n""«»"" ') + r"]+$"), "", text).strip()
        return text

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

        if word_alignment:
            return phonemized_string, (words, phonemes_list)
        return phonemized_string

    def tokenize(self, text: str) -> list[int]:
        return [self.word_index_dictionary[c] for c in text if c in self.word_index_dictionary]

    def detokenize(self, tokens: list[int]) -> str:
        return "".join(self.index_to_symbol[t] for t in tokens if t in self.index_to_symbol)

    def __call__(self, text: str) -> list[int]:
        return self.tokenize(text)
