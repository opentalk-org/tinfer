use crate::utf8::utf8_next;

fn is_number_sep_byte(c: u8) -> bool {
    c == b'.' || c == b',' || c == b':' || c == b'/' || c == b'-'
}

fn is_number_sep_token(tok: &str) -> bool {
    if tok.len() == 1 && is_number_sep_byte(tok.as_bytes()[0]) {
        return true;
    }
    // handling of em-dash, en-dash
    let bi = 0usize;
    if bi >= tok.as_bytes().len() {
        return false;
    }
    let (cp, _) = utf8_next(tok.as_bytes(), bi);
    cp == 0x2013 || cp == 0x2014
}

fn is_space_cp(cp: u32) -> bool {
    cp == (' ' as u32)
        || cp == ('\t' as u32)
        || cp == ('\n' as u32)
        || cp == ('\r' as u32)
        || cp == ('\x0C' as u32)
        || cp == ('\x0B' as u32)
}

fn is_word_cp(cp: u32) -> bool {
    if cp < 128 {
        return (b'a' as u32) <= cp && cp <= (b'z' as u32)
            || (b'A' as u32) <= cp && cp <= (b'Z' as u32)
            || (b'0' as u32) <= cp && cp <= (b'9' as u32)
            || cp == (b'_' as u32);
    }
    if cp == 0xB0
        || cp == 0xA1
        || cp == 0xBF
        || cp == 0xAB
        || cp == 0xBB
        || cp == 0x2014
        || cp == 0x2026
    {
        return false;
    }
    if (0x2000..=0x206F).contains(&cp) {
        return false;
    }
    true
}

pub fn tokenize(text: &str) -> Vec<String> {
    let bytes = text.as_bytes();
    let mut raw: Vec<String> = Vec::new();

    let mut i = 0usize;
    while i < bytes.len() {
        let start = i;
        let (cp, next) = utf8_next(bytes, i);
        let is_space = is_space_cp(cp);
        let is_word = is_word_cp(cp);
        i = next;

        if is_space {
            while i < bytes.len() {
                let (cp2, n2) = utf8_next(bytes, i);
                if !is_space_cp(cp2) {
                    break;
                }
                i = n2;
            }
            raw.push(text[start..i].to_owned());
        } else if is_word {
            while i < bytes.len() {
                let (cp2, n2) = utf8_next(bytes, i);
                if !is_word_cp(cp2) {
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
            if !is_space_cp(cp) {
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
            if cp < (b'0' as u32) || cp > (b'9' as u32) {
                is_digit_block = false;
                break;
            }
            bi2 = n;
        }

        let is_number_sep_run = is_number_sep_token(&tok);

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

