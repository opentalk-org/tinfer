use std::collections::{HashMap, HashSet};
use std::sync::Mutex;

use espeak_align_core::{AlignmentSpan, Engine};

use crate::{Error, Result};

const PUNCTUATION: &str = ";:,.!?¡¿—–…\"«»“”";

pub(super) struct PreparedText {
    pub phonemes: String,
    pub mapping: Vec<MappedItem>,
}

pub(super) struct MappedItem {
    pub original_start: usize,
    pub original_end: usize,
    pub phoneme_count: usize,
}

pub(super) fn prepare_text(
    text: &str,
    phonemized: bool,
    language: &str,
    symbols: &[String],
    phonemizers: &Mutex<HashMap<String, Engine>>,
) -> Result<PreparedText> {
    if phonemized {
        let phonemes = filter_to_vocab(text.trim(), symbols);
        return Ok(PreparedText {
            mapping: vec![MappedItem { original_start: 0, original_end: text.len(), phoneme_count: phonemes.chars().count() }],
            phonemes,
        });
    }
    let (normalized, spans) = normalize_with_mapping(text);
    let mut engines = phonemizers.lock().expect("StyleTTS2 phonemizer lock poisoned");
    let engine = engines.entry(language.to_owned()).or_insert_with(|| Engine::new(language, true, 4));
    let aligned = engine.align_with_spans(&normalized, PUNCTUATION, 8).map_err(|error| Error::Validation(error.to_string()))?;
    Ok(mapped_text(text, &spans, aligned, symbols))
}

pub(super) fn tokenize<S: AsRef<str>>(text: &str, symbols: &[S]) -> Vec<i64> {
    let vocabulary = symbols
        .iter()
        .enumerate()
        .filter_map(|(index, symbol)| symbol.as_ref().chars().next().map(|character| (character, index as i64)))
        .collect::<HashMap<_, _>>();
    let mut tokens = vec![0];
    tokens.extend(text.chars().filter_map(|character| vocabulary.get(&character).copied()));
    tokens
}

#[cfg(test)]
pub(super) fn normalize_text(text: &str) -> String {
    normalize_with_mapping(text).0
}

fn normalize_with_mapping(text: &str) -> (String, Vec<(usize, usize)>) {
    let mut normalized = String::new();
    let mut spans = Vec::new();
    let mut characters = text.char_indices().peekable();
    while let Some((start, character)) = characters.next() {
        let end = start + character.len_utf8();
        if character.is_whitespace() {
            let mut whitespace_end = end;
            while let Some((next, value)) = characters.peek().copied() {
                if !value.is_whitespace() {
                    break;
                }
                whitespace_end = next + value.len_utf8();
                characters.next();
            }
            if !normalized.is_empty() {
                normalized.push(' ');
                spans.push((start, whitespace_end));
            }
            continue;
        }
        let replacement = match character {
            '(' | ')' | '*' | '/' | '[' | ']' => continue,
            '„' | '“' | '”' | '«' | '»' => '"',
            '-' | '−' | '‒' | '–' => '—',
            character => character,
        };
        normalized.push(replacement);
        spans.push((start, end));
    }
    if normalized.ends_with(' ') {
        normalized.pop();
        spans.pop();
    }
    if !matches!(normalized.chars().last(), Some('.' | '?' | '!')) {
        normalized.push('.');
        spans.push((text.len(), text.len()));
    }
    (normalized, spans)
}

fn filter_to_vocab(text: &str, symbols: &[String]) -> String {
    let vocabulary = symbols.iter().filter_map(|symbol| symbol.chars().next()).collect::<HashSet<_>>();
    text.chars().filter(|character| vocabulary.contains(character)).collect()
}

fn mapped_text(original: &str, spans: &[(usize, usize)], aligned: Vec<AlignmentSpan>, symbols: &[String]) -> PreparedText {
    let mut filtered = aligned
        .into_iter()
        .map(|item| (item.start, item.end, filter_to_vocab(&item.phonemes.split_whitespace().collect::<Vec<_>>().join(" "), symbols)))
        .collect::<Vec<_>>();
    let nonempty = filtered.iter().filter(|(_, _, phonemes)| !phonemes.is_empty()).count();
    let mut seen = 0;
    let mut consumed = 0;
    let mut phonemes = String::new();
    let mut mapping = Vec::new();
    for (start, end, item_phonemes) in &mut filtered {
        let original_start = spans.get(*start).map_or(consumed, |span| span.0);
        let original_end = end.checked_sub(1).and_then(|index| spans.get(index)).map_or(original_start, |span| span.1);
        if consumed < original_start {
            mapping.push(MappedItem { original_start: consumed, original_end: original_start, phoneme_count: 0 });
        }
        if !item_phonemes.is_empty() {
            seen += 1;
            if seen < nonempty && symbols.iter().any(|symbol| symbol == " ") {
                item_phonemes.push(' ');
            }
        }
        let phoneme_count = item_phonemes.chars().count();
        phonemes.push_str(item_phonemes);
        mapping.push(MappedItem { original_start, original_end, phoneme_count });
        consumed = consumed.max(original_end);
    }
    if consumed < original.len() {
        mapping.push(MappedItem { original_start: consumed, original_end: original.len(), phoneme_count: 0 });
    }
    PreparedText { phonemes, mapping }
}
