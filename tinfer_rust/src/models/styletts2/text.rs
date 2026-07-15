use std::collections::{BTreeMap, HashMap, HashSet};
use std::sync::Mutex;

use espeak_align_core::{AlignmentSpan, Engine};

use crate::{Error, Result};

const PUNCTUATION: &str = ";:,.!?¡¿—–…\"«»“”";

#[derive(Clone)]
pub(super) struct PreparedText {
    pub phonemes: String,
    pub mapping: Vec<MappedItem>,
}

#[derive(Clone)]
pub(super) struct MappedItem {
    pub original_start: usize,
    pub original_end: usize,
    pub phoneme_count: usize,
}

pub(super) fn prepare_texts(
    texts: &[&str],
    phonemized: &[bool],
    languages: &[&str],
    normalization: &[&str],
    symbols: &[String],
    phonemizers: &Mutex<HashMap<String, Engine>>,
) -> Result<Vec<PreparedText>> {
    assert_eq!(texts.len(), phonemized.len(), "text preparation flags must match the batch");
    assert_eq!(texts.len(), languages.len(), "text preparation languages must match the batch");
    assert_eq!(texts.len(), normalization.len(), "text normalization modes must match the batch");
    let mut output = vec![None; texts.len()];
    let mut normalized = vec![None; texts.len()];
    let mut groups = BTreeMap::<&str, Vec<usize>>::new();
    for index in 0..texts.len() {
        if phonemized[index] {
            let phonemes = filter_to_vocab(texts[index].trim(), symbols);
            output[index] = Some(PreparedText {
                mapping: vec![MappedItem { original_start: 0, original_end: texts[index].len(), phoneme_count: phonemes.chars().count() }],
                phonemes,
            });
        } else {
            normalized[index] = Some(if normalization[index] == "off" {
                identity_with_mapping(texts[index])
            } else {
                normalize_with_mapping(texts[index])
            });
            groups.entry(languages[index]).or_default().push(index);
        }
    }
    let mut engines = phonemizers.lock().expect("StyleTTS2 phonemizer lock poisoned");
    for (language, indices) in groups {
        let batch =
            indices.iter().map(|index| normalized[*index].as_ref().expect("unphonemized text is normalized").0.clone()).collect::<Vec<_>>();
        let engine = engines.entry(language.to_owned()).or_insert_with(|| Engine::new(language, true, 16));
        let aligned = engine.align_batch_with_spans(&batch, PUNCTUATION, 16).map_err(|error| Error::Validation(error.to_string()))?;
        for (index, items) in indices.into_iter().zip(aligned) {
            let (_, spans) = normalized[index].take().expect("unphonemized text is normalized");
            output[index] = Some(mapped_text(texts[index], &spans, items, symbols));
        }
    }
    Ok(output.into_iter().map(|item| item.expect("every text is prepared")).collect())
}

fn identity_with_mapping(text: &str) -> (String, Vec<(usize, usize)>) {
    (text.to_owned(), text.char_indices().map(|(start, character)| (start, start + character.len_utf8())).collect())
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
