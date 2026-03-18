use crate::utf8::utf8_next;
use crate::char_match;

pub fn tokenize(text: &str) -> Vec<String> {
    let bytes = text.as_bytes();
    let mut raw: Vec<String> = Vec::new();

    let mut i = 0usize;
    while i < bytes.len() {
        let start = i;
        let (cp, next) = utf8_next(bytes, i);
        let is_space = char_match::is_space_cp(cp);
        let is_word = char_match::is_word_cp(cp);
        i = next;

        if is_space {
            while i < bytes.len() {
                let (cp2, n2) = utf8_next(bytes, i);
                if !char_match::is_space_cp(cp2) {
                    break;
                }
                i = n2;
            }
            raw.push(text[start..i].to_owned());
        } else if is_word {
            while i < bytes.len() {
                let (cp2, n2) = utf8_next(bytes, i);
                if !char_match::is_word_cp(cp2) {
                    break;
                }
                i = n2;
            }
            raw.push(text[start..i].to_owned());
        } else {
            raw.push(text[start..i].to_owned());
        }
    }

    if raw.is_empty() {
        return Vec::new();
    }

    let mut tokens: Vec<String> = Vec::new();
    let mut pending_space = String::new();
    let mut in_number = false;
    let mut number_buf = String::new();

    for tok in raw {
        let tok_bytes = tok.as_bytes();

        let mut all_space = true;
        let mut bi = 0usize;
        while bi < tok_bytes.len() {
            let (cp, n) = utf8_next(tok_bytes, bi);
            if !char_match::is_space_cp(cp) {
                all_space = false;
                break;
            }
            bi = n;
        }

        if all_space {
            if in_number {
                number_buf.push_str(&tok);
            } else if let Some(last) = tokens.last_mut() {
                last.push_str(&tok);
            } else {
                pending_space.push_str(&tok);
            }
            continue;
        }

        let mut is_digit_block = true;
        let mut bi2 = 0usize;
        while bi2 < tok_bytes.len() {
            let (cp, n) = utf8_next(tok_bytes, bi2);
            if cp >= 128 {
                is_digit_block = false;
                break;
            }
            if !char_match::is_digit_cp(cp) {
                is_digit_block = false;
                break;
            }
            bi2 = n;
        }

        let is_number_sep_run = char_match::is_number_sep_token(&tok);

        if is_digit_block || (in_number && is_number_sep_run) {
            if !in_number {
                if pending_space.is_empty() {
                    number_buf = tok;
                } else {
                    number_buf = pending_space.clone();
                    number_buf.push_str(&tok);
                    pending_space.clear();
                }
                in_number = true;
            } else {
                number_buf.push_str(&tok);
            }
        } else {
            if in_number {
                tokens.push(number_buf.clone());
                number_buf.clear();
                in_number = false;
            }
            let mut t = if pending_space.is_empty() {
                tok
            } else {
                let mut s = pending_space.clone();
                s.push_str(&tok);
                pending_space.clear();
                s
            };
            tokens.push(std::mem::take(&mut t));
        }
    }

    if in_number && !number_buf.is_empty() {
        tokens.push(number_buf);
    }

    tokens
}

