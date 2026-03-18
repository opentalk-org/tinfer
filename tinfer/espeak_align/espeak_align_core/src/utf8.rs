pub fn utf8_len(first: u8) -> usize {
    if first < 0x80 {
        1
    } else if (first & 0xE0) == 0xC0 {
        2
    } else if (first & 0xF0) == 0xE0 {
        3
    } else if (first & 0xF8) == 0xF0 {
        4
    } else { // should not happen
        1
    }
}

pub fn utf8_next(bytes: &[u8], i: usize) -> (u32, usize) {
    if i >= bytes.len() {
        return (0, i);
    }

    let c = bytes[i];
    let len = utf8_len(c);
    if i + len > bytes.len() { // invalid UTF-8 sequence
        return (c as u32, i + 1);
    }

    let cp = match len {
        1 => c as u32,
        2 => (((c & 0x1F) as u32) << 6) | ((bytes[i + 1] & 0x3F) as u32),
        3 => {
            (((c & 0x0F) as u32) << 12)
                | (((bytes[i + 1] & 0x3F) as u32) << 6)
                | ((bytes[i + 2] & 0x3F) as u32)
        }
        _ => { // 4 bytes
            (((c & 0x07) as u32) << 18)
                | (((bytes[i + 1] & 0x3F) as u32) << 12)
                | (((bytes[i + 2] & 0x3F) as u32) << 6)
                | ((bytes[i + 3] & 0x3F) as u32)
        }
    };

    (cp, i + len)
}

