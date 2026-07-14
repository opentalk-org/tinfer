use std::ops::Range;

use pysbd::{Options, Segmenter};

use crate::{Error, Result};

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct PreparedChunk {
    pub text: String,
    pub span: Range<usize>,
    pub bytes: usize,
}

#[derive(Clone, Copy)]
struct Limits {
    trigger: usize,
    no_split: usize,
}

pub(crate) struct Chunker {
    segmenter: Segmenter,
}

impl Chunker {
    pub fn new(language: &str) -> Result<Self> {
        let code = language.split(['-', '_']).next().expect("language is non-empty");
        let segmenter =
            Segmenter::new(code, Options { clean: false, doc_type: None }).map_err(|error| Error::Validation(error.to_string()))?;
        Ok(Self { segmenter })
    }

    pub fn prepare(&self, text: &str, offset: usize, index: usize, schedule: &[usize]) -> Result<Vec<PreparedChunk>> {
        let chunks = if text.chars().count() <= limits(schedule, index).no_split {
            vec![text.to_owned()]
        } else {
            let mut pieces = Vec::new();
            for sentence in self.sentence_chunks(text) {
                pieces.extend(self.split_oversized(&sentence, schedule, index + pieces.len()));
            }
            pack(pieces, schedule, index)
        };
        source_spans(text, chunks, offset)
    }

    fn sentence_chunks(&self, text: &str) -> Vec<String> {
        let Ok(sentences) = self.segmenter.segment(text) else {
            return vec![text.to_owned()];
        };
        let mut chunks = Vec::new();
        let mut cursor = 0;
        for sentence in sentences {
            let Some(relative) = text[cursor..].find(&sentence) else {
                return vec![text.to_owned()];
            };
            let start = cursor + relative;
            if start > cursor {
                chunks.push(text[cursor..start].to_owned());
            }
            cursor = start + sentence.len();
            chunks.push(text[start..cursor].to_owned());
        }
        if cursor < text.len() {
            chunks.push(text[cursor..].to_owned());
        }
        chunks.into_iter().filter(|chunk| !chunk.is_empty()).collect()
    }

    fn split_oversized(&self, text: &str, schedule: &[usize], index: usize) -> Vec<String> {
        let mut pending = vec![text.to_owned()];
        for separator in [Separator::DoubleNewline, Separator::Newline, Separator::Sentence, Separator::Clause, Separator::Space] {
            let mut next = Vec::new();
            let mut changed = false;
            for part in pending {
                if part.chars().count() <= limits(schedule, index + next.len()).trigger {
                    next.push(part);
                    continue;
                }
                let split = separator.split(&part);
                changed |= split.len() > 1;
                next.extend(split);
            }
            pending = if changed { pack(next, schedule, index) } else { next };
        }
        let mut chunks = Vec::new();
        for part in pending {
            if part.chars().count() <= limits(schedule, index + chunks.len()).trigger {
                chunks.push(part);
            } else {
                chunks.extend(split_by_limits(&part, schedule, index + chunks.len()));
            }
        }
        chunks.into_iter().filter(|chunk| !chunk.is_empty()).collect()
    }
}

#[derive(Clone, Copy)]
enum Separator {
    DoubleNewline,
    Newline,
    Sentence,
    Clause,
    Space,
}

impl Separator {
    fn split(self, text: &str) -> Vec<String> {
        match self {
            Self::DoubleNewline => split_literal(text, "\n\n"),
            Self::Newline => split_literal(text, "\n"),
            Self::Sentence => split_after(text, ".!?"),
            Self::Clause => split_after(text, ",;"),
            Self::Space => split_words(text),
        }
    }
}

fn limits(schedule: &[usize], index: usize) -> Limits {
    let schedule_index = index.min(schedule.len() - 1);
    let trigger = schedule[schedule_index];
    let no_split = if let Some(next) = schedule.get(schedule_index + 1) {
        *next
    } else if schedule.len() > 1 {
        trigger + (trigger - schedule[schedule.len() - 2]).max(1)
    } else {
        trigger + (trigger / 3).max(1)
    };
    Limits { trigger, no_split }
}

fn split_literal(text: &str, separator: &str) -> Vec<String> {
    let mut output = Vec::new();
    let mut cursor = 0;
    while let Some(relative) = text[cursor..].find(separator) {
        let end = cursor + relative + separator.len();
        output.push(text[cursor..end].to_owned());
        cursor = end;
    }
    if cursor < text.len() {
        output.push(text[cursor..].to_owned());
    }
    output
}

fn split_after(text: &str, punctuation: &str) -> Vec<String> {
    let mut output = Vec::new();
    let mut start = 0;
    let mut chars = text.char_indices().peekable();
    while let Some((_, character)) = chars.next() {
        if !punctuation.contains(character) || !chars.peek().is_some_and(|(_, next)| next.is_whitespace()) {
            continue;
        }
        let mut end = chars.peek().map_or(text.len(), |(byte, _)| *byte);
        while let Some((byte, next)) = chars.peek().copied() {
            if !next.is_whitespace() {
                break;
            }
            chars.next();
            end = byte + next.len_utf8();
        }
        output.push(text[start..end].to_owned());
        start = end;
    }
    if start < text.len() {
        output.push(text[start..].to_owned());
    }
    output
}

fn split_words(text: &str) -> Vec<String> {
    let mut output = Vec::new();
    let mut start = 0;
    let mut in_space = false;
    for (byte, character) in text.char_indices() {
        if in_space && !character.is_whitespace() {
            output.push(text[start..byte].to_owned());
            start = byte;
        }
        in_space = character.is_whitespace();
    }
    if start < text.len() {
        output.push(text[start..].to_owned());
    }
    output
}

fn split_by_limits(text: &str, schedule: &[usize], index: usize) -> Vec<String> {
    let mut output = Vec::new();
    let mut remaining = text;
    while !remaining.is_empty() {
        let current = limits(schedule, index + output.len());
        if remaining.chars().count() <= current.no_split {
            output.push(remaining.to_owned());
            break;
        }
        let byte = remaining.char_indices().nth(current.trigger).map_or(remaining.len(), |(byte, _)| byte);
        output.push(remaining[..byte].to_owned());
        remaining = &remaining[byte..];
    }
    output
}

fn pack(pieces: Vec<String>, schedule: &[usize], index: usize) -> Vec<String> {
    let mut chunks = Vec::new();
    let mut current = String::new();
    for piece in pieces.into_iter().filter(|piece| !piece.is_empty()) {
        let trigger = limits(schedule, index + chunks.len()).trigger;
        if !current.is_empty() && current.chars().count() + piece.chars().count() > trigger {
            chunks.push(std::mem::replace(&mut current, piece));
        } else {
            current.push_str(&piece);
        }
    }
    if !current.is_empty() {
        chunks.push(current);
    }
    if chunks.len() > 1 {
        let previous = index + chunks.len() - 2;
        if chunks[chunks.len() - 2].chars().count() + chunks.last().expect("two chunks").chars().count()
            <= limits(schedule, previous).no_split
        {
            let last = chunks.pop().expect("two chunks");
            chunks.last_mut().expect("one chunk remains").push_str(&last);
        }
    }
    chunks.into_iter().filter(|chunk| !chunk.trim().is_empty()).collect()
}

fn source_spans(text: &str, chunks: Vec<String>, offset: usize) -> Result<Vec<PreparedChunk>> {
    let mut prepared: Vec<PreparedChunk> = Vec::new();
    let mut byte_cursor = 0;
    for chunk in chunks {
        let relative =
            text[byte_cursor..].find(&chunk).ok_or_else(|| Error::Validation("text chunker output does not match source text".into()))?;
        let byte_start = byte_cursor + relative;
        if let Some(previous) = prepared.last_mut() {
            previous.span.end = offset + text[..byte_start].chars().count();
            previous.bytes = byte_start;
        }
        byte_cursor = byte_start + chunk.len();
        prepared.push(PreparedChunk {
            text: chunk,
            span: offset + text[..byte_start].chars().count()..offset + text[..byte_cursor].chars().count(),
            bytes: byte_cursor,
        });
    }
    let last = prepared.last_mut().ok_or_else(|| Error::Validation("text chunker produced an empty chunk".into()))?;
    last.span.end = offset + text.chars().count();
    last.bytes = text.len();
    Ok(prepared)
}
