use crate::utf8::utf8_next;
use std::collections::BTreeSet;

fn is_digit_cp(cp: u32) -> bool {
    (b'0' as u32) <= cp && cp <= (b'9' as u32)
}

fn is_sep_cp(cp: u32) -> bool {
    cp == (b'.' as u32)
        || cp == (b',' as u32)
        || cp == (b':' as u32)
        || cp == (b'/' as u32)
        || cp == (b'-' as u32)
        || cp == 0x2013
        || cp == 0x2014
}

fn punct_set_from_string(punctuation: &str) -> BTreeSet<u32> {
    let mut out = BTreeSet::new();
    let bytes = punctuation.as_bytes();
    let mut i = 0usize;
    while i < bytes.len() {
        let (cp, next) = utf8_next(bytes, i);
        out.insert(cp);
        i = next;
    }
    out
}

pub fn split_by_punctuation_impl(
    text: &str,
    punctuation: &str,
) -> (Vec<String>, Vec<(i32, String)>) {
    let bytes = text.as_bytes();

    let mut cps: Vec<u32> = Vec::new();
    let mut byte_start: Vec<usize> = Vec::new();

    let mut i = 0usize;
    while i < bytes.len() {
        byte_start.push(i);
        let (cp, next) = utf8_next(bytes, i);
        cps.push(cp);
        i = next;
    }
    byte_start.push(bytes.len());

    let n = cps.len();

    let mut protected_char_pos: BTreeSet<usize> = BTreeSet::new();
    let mut ci = 0usize;
    while ci < n {
        if !is_digit_cp(cps[ci]) {
            ci += 1;
            continue;
        }

        let start = ci;
        while ci < n && (is_digit_cp(cps[ci]) || is_sep_cp(cps[ci])) {
            ci += 1;
        }
        while ci > start && !is_digit_cp(cps[ci - 1]) {
            ci -= 1;
        }

        if ci > start {
            protected_char_pos.extend(start..ci);
        }
    }

    let punct_set = punct_set_from_string(punctuation);

    let mut chunks_out: Vec<String> = Vec::new();
    let mut puncts_out: Vec<(i32, String)> = Vec::new();

    let mut last_byte = 0usize;
    let mut idx: i32 = 0;

    for pos in 0..n {
        if protected_char_pos.contains(&pos) {
            continue;
        }
        if !punct_set.contains(&cps[pos]) {
            continue;
        }

        let byte_end = byte_start[pos + 1];
        let chunk = &text[last_byte..byte_start[pos]];
        if !chunk.is_empty() {
            chunks_out.push(chunk.to_owned());
            idx += 1;
        }

        let chunk_idx = if idx > 0 { idx - 1 } else { 0 };
        let mark = &text[byte_start[pos]..byte_end];
        puncts_out.push((chunk_idx, mark.to_owned()));
        last_byte = byte_end;
    }

    let tail = &text[last_byte..];
    if !tail.is_empty() {
        chunks_out.push(tail.to_owned());
    }

    (chunks_out, puncts_out)
}

