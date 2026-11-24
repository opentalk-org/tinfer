use crate::{Result, TextSpan};

pub(crate) fn python_whitespace(character: char) -> bool {
    character.is_whitespace() || matches!(character, '\u{1c}'..='\u{1f}')
}

pub(crate) fn locate_spans(original: &str, sentences: &[String]) -> Result<Vec<TextSpan>> {
    let mut spans = Vec::with_capacity(sentences.len());
    let mut prior_end = 0;
    for sentence in sentences {
        // Upstream searches the entire source for each sentence and accepts
        // matches ending after the previous one, including overlapping matches.
        // It omits sentences altered by internal marker restoration when they
        // no longer occur in the source; preserving that behavior is required
        // for Python parity.
        let mut cursor = 0;
        while let Some(relative) = original[cursor..].find(sentence) {
            let start = cursor + relative;
            let mut end = start + sentence.len();
            while let Some(character) = original[end..].chars().next() {
                if !python_whitespace(character) {
                    break;
                }
                end += character.len_utf8();
            }
            if end > prior_end {
                spans.push(TextSpan {
                    text: original[start..end].to_owned(),
                    start: original[..start].chars().count(),
                    end: original[..end].chars().count(),
                });
                prior_end = end;
                break;
            }
            cursor = end;
            if sentence.is_empty() && end == start {
                cursor += original[cursor..].chars().next().map_or(0, char::len_utf8);
                if cursor == end {
                    break;
                }
            }
        }
    }
    Ok(spans)
}
