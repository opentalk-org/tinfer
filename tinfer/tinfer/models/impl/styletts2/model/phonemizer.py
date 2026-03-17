from __future__ import annotations
from typing import Optional, List, Union, Tuple
import threading
import re
import phonemizer
from nltk.tokenize import TweetTokenizer


_espeak_backend_cache = {}
_cache_lock = threading.Lock()


def get_espeak_backend(language: str, preserve_punctuation: bool, with_stress: bool, tie: bool):
    key = (language, preserve_punctuation, with_stress, tie)
    with _cache_lock:
        if key not in _espeak_backend_cache:
            _espeak_backend_cache[key] = {
                'backend': phonemizer.backend.EspeakBackend(
                    language=language,
                    preserve_punctuation=preserve_punctuation,
                    with_stress=with_stress,
                    tie=tie,
                ),
                'lock': threading.Lock()
            }
        return _espeak_backend_cache[key]


class StyleTTS2Phonemizer:
    def __init__(
        self,
        language: str = "pl",
        preserve_punctuation: bool = True,
        with_stress: bool = True,
        tie: bool = True
    ):
        self.language = language
        self.preserve_punctuation = preserve_punctuation
        self.with_stress = with_stress
        self.tie = tie
        self.phonemizer = None
        self._phonemizer_lock = None
        self.t_tokenizer = TweetTokenizer()
        self._init_phonemizer()
        self._init_tokenizer()
    
    def _init_phonemizer(self):
        backend_info = get_espeak_backend(
            language=self.language,
            preserve_punctuation=self.preserve_punctuation,
            with_stress=self.with_stress,
            tie=self.tie,
        )
        self.phonemizer = backend_info['backend']
        self._phonemizer_lock = backend_info['lock']
    
    def _init_tokenizer(self):
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
        return text

    def phonemize(self, text: str, separator: Optional[phonemizer.separator.Separator] = None) -> str:        
        try:
            with self._phonemizer_lock: # TODO: espeak phonemizer is not thread safe
                if separator is not None:
                    phonemized = self.phonemizer.phonemize([text], separator=separator)[0]
                    phonemized = phonemized.replace('|w|', ' |f|') # not sure why espeak-ng doesn't handle it
                else:
                    phonemized = self.phonemizer.phonemize([text])[0]
                    phonemized = phonemized.replace(' w ', ' f ')
            
            return phonemized
        except Exception as e:
            return text
    
    def process_text(self, text: str, phonemize: bool = True, word_alignment: bool = False) -> Union[str, Tuple[str, str]]:
        text = text.strip()
        
        if not phonemize:
            return text
        
        preprocessed = self._preprocess_text(text)
        tokenized = " ".join(self.t_tokenizer.tokenize(preprocessed))
        
        if word_alignment:
            separator = phonemizer.separator.Separator(word="|")
            ps_with_sep = self.phonemize(tokenized, separator=separator)
            ps_with_sep = ps_with_sep.replace('``', '"')
            ps_with_sep = ps_with_sep.replace("''", '"')
            
            ps_for_inference = ps_with_sep.replace('|', ' ')
            ps_for_inference = ' '.join(ps_for_inference.split())

            return ps_for_inference, ps_with_sep
        else:
            ps = self.phonemize(tokenized)
            ps = ps.replace('``', '"')
            ps = ps.replace("''", '"')
            return ps
    
    def tokenize(self, text: str) -> List[int]:
        return [self.word_index_dictionary[char] for char in text if char in self.word_index_dictionary]
    
    def detokenize(self, tokens: List[int]) -> str:
        return ''.join(self.index_to_symbol[token] for token in tokens if token in self.index_to_symbol)
    
    def __call__(self, text: str) -> List[int]:
        return self.tokenize(text)

