use crate::{Error, Result, TextSpan};

pub(crate) fn locate_spans(original: &str, sentences: &[String]) -> Result<Vec<TextSpan>> {
    let mut spans = Vec::with_capacity(sentences.len());
    let mut byte_cursor = 0;
    for sentence in sentences {
        let relative = original[byte_cursor..]
            .find(sentence)
            .ok_or_else(|| Error::Span { sentence: sentence.clone(), offset: original[..byte_cursor].chars().count() })?;
        let byte_start = byte_cursor + relative;
        byte_cursor = byte_start + sentence.len();
        while let Some(character) = original[byte_cursor..].chars().next() {
            if !character.is_whitespace() {
                break;
            }
            byte_cursor += character.len_utf8();
        }
        spans.push(TextSpan {
            text: original[byte_start..byte_cursor].to_owned(),
            start: original[..byte_start].chars().count(),
            end: original[..byte_cursor].chars().count(),
        });
    }

    if byte_cursor != original.len() {
        return Err(Error::Span { sentence: original[byte_cursor..].to_owned(), offset: original[..byte_cursor].chars().count() });
    }
    Ok(spans)
}
