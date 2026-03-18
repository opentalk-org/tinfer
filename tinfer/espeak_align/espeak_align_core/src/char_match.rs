use crate::utf8::utf8_next;

pub fn is_space_cp(cp: u32) -> bool {
    cp == (' ' as u32)
        || cp == ('\t' as u32)
        || cp == ('\n' as u32)
        || cp == ('\r' as u32)
        || cp == ('\x0C' as u32)
        || cp == ('\x0B' as u32)
}

pub fn is_digit_cp(cp: u32) -> bool {
    (b'0' as u32) <= cp && cp <= (b'9' as u32)
}

pub fn is_number_sep_byte(c: u8) -> bool {
    c == b'.' || c == b',' || c == b':' || c == b'/' || c == b'-'
}

pub fn is_number_sep_cp(cp: u32) -> bool {
    cp == (b'.' as u32)
        || cp == (b',' as u32)
        || cp == (b':' as u32)
        || cp == (b'/' as u32)
        || cp == (b'-' as u32)
        || cp == 0x2013
        || cp == 0x2014
}

pub fn is_number_sep_token(tok: &str) -> bool {
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

pub fn is_word_cp(cp: u32) -> bool {
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

