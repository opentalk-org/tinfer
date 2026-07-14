use regex::{Captures, Regex};

use crate::{Error, ListMode, Result};

const ROMAN: &[&str] =
    &["i", "ii", "iii", "iv", "v", "vi", "vii", "viii", "ix", "x", "xi", "xii", "xiii", "xiv", "xv", "xvi", "xvii", "xviii", "xix", "xx"];

pub(crate) fn add_line_breaks(text: &str, mode: ListMode) -> Result<String> {
    let bullet = Regex::new(r"\s([•⁃]\s*\d{1,2}\.)").map_err(regex_error)?;
    let output = bullet.replace_all(text, "\r$1").into_owned();
    let attached = Regex::new(r"(⁃)(\d{1,2})\.(\s)").map_err(regex_error)?;
    let output = attached.replace_all(&output, "$1$2∯$3").into_owned();
    let output = match mode {
        ListMode::Standard => protect_letter_lists(&output, false)?,
        ListMode::NoAlphabetical => output,
    };
    let output = protect_letter_lists(&output, true)?;
    let output = protect_numbered_lists(&output, '.')?;
    protect_numbered_lists(&output, ')')
}

pub(crate) fn protect_roman_parens(text: &str) -> Result<String> {
    let regex = Regex::new(r"(?i)\(([mdclxvi]+)\)(\s[A-Z])").map_err(regex_error)?;
    Ok(regex.replace_all(text, "&✂&$1&⌬&$2").into_owned())
}

fn protect_numbered_lists(text: &str, delimiter: char) -> Result<String> {
    let pattern = if delimiter == '.' { r"(?m)(^|\s)(\d{1,2})(\.)(\s|\))" } else { r"(?m)(^|\s)(\d{1,2})(\))(\s)" };
    let regex = Regex::new(pattern).map_err(regex_error)?;
    let numbers: Vec<u8> = regex
        .captures_iter(text)
        .map(|caps| caps[2].parse::<u8>())
        .collect::<std::result::Result<_, _>>()
        .map_err(|error| Error::Regex { pattern: pattern.into(), message: error.to_string() })?;
    let selected = sequence_members(&numbers);
    if !selected.iter().any(|value| *value) {
        return Ok(text.to_owned());
    }
    let suppress_breaks = delimiter == '.' && Regex::new(r"(?i)\bfor\s\d{1,2}\.\s[a-z]").map_err(regex_error)?.is_match(text);
    let replacement = if delimiter == '.' { "∯" } else { "" };
    let mut first = true;
    let mut index = 0;
    Ok(regex
        .replace_all(text, |caps: &Captures<'_>| {
            let replace = selected[index];
            index += 1;
            if !replace {
                return caps[0].to_owned();
            }
            let separator = &caps[1];
            let bullet =
                caps.get(0).and_then(|found| text[..found.start()].chars().next_back()).is_some_and(|character| "•⁃-".contains(character));
            let at_start = caps.get(0).is_some_and(|found| text[..found.start()].trim().is_empty());
            let line = if suppress_breaks || bullet || (first && at_start) || separator.contains(['\n', '\r']) {
                separator.to_owned()
            } else {
                "\r".into()
            };
            first = false;
            let delimiter = if delimiter == ')' { ")" } else { replacement };
            format!("{line}{}{delimiter}{}", &caps[2], &caps[4])
        })
        .into_owned())
}

fn protect_letter_lists(text: &str, roman: bool) -> Result<String> {
    let regex = Regex::new(r"(?im)(^|\s)(\(?)([a-z]+)(\.?\)|\.)").map_err(regex_error)?;
    let items: Vec<String> = regex
        .captures_iter(text)
        .filter_map(|caps| {
            let item = caps[3].to_owned();
            let followed_by_space = caps.get(0).is_some_and(|found| text[found.end()..].starts_with(char::is_whitespace));
            let valid = followed_by_space
                && item.chars().all(|character| character.is_ascii_lowercase())
                && if roman { ROMAN.contains(&item.as_str()) } else { item.len() == 1 };
            valid.then_some(item)
        })
        .collect();
    let selected = letter_sequence_members(&items, roman);
    if !selected.iter().any(|value| *value) {
        return Ok(text.to_owned());
    }
    let mut first = true;
    let mut valid_index = 0;
    Ok(regex
        .replace_all(text, |caps: &Captures<'_>| {
            let item = caps[3].to_owned();
            let followed_by_space = caps.get(0).is_some_and(|found| text[found.end()..].starts_with(char::is_whitespace));
            let valid = followed_by_space
                && item.chars().all(|character| character.is_ascii_lowercase())
                && if roman { ROMAN.contains(&item.as_str()) } else { item.len() == 1 };
            if !valid {
                return caps[0].to_owned();
            }
            let replace = selected[valid_index];
            valid_index += 1;
            if !replace {
                return caps[0].to_owned();
            }
            let at_start = caps.get(0).is_some_and(|found| text[..found.start()].trim().is_empty());
            let prefix = if (first && at_start) || caps[1].contains(['\n', '\r']) { caps[1].to_owned() } else { "\r".into() };
            first = false;
            let left = if &caps[2] == "(" { "&✂&" } else { "" };
            let suffix = match &caps[4] {
                "." => "∯",
                ".)" => "∯)",
                _ => ")",
            };
            format!("{prefix}{left}{}{suffix}", &caps[3])
        })
        .into_owned())
}

fn sequence_members(numbers: &[u8]) -> Vec<bool> {
    (0..numbers.len())
        .map(|index| {
            let adjacent = |left: u8, right: u8| right == left + 1 || (left, right) == (9, 0) || (left, right) == (0, 9);
            index.checked_sub(1).is_some_and(|prior| adjacent(numbers[prior], numbers[index]))
                || numbers.get(index + 1).is_some_and(|next| adjacent(numbers[index], *next))
        })
        .collect()
}

fn letter_sequence_members(items: &[String], roman: bool) -> Vec<bool> {
    let adjacent = |left: &str, right: &str| {
        let index = |item: &str| {
            if roman { ROMAN.iter().position(|value| *value == item) } else { item.bytes().next().map(|value| (value - b'a') as usize) }
        };
        matches!((index(left), index(right)), (Some(left), Some(right)) if left.abs_diff(right) == 1)
    };
    (0..items.len())
        .map(|index| {
            index.checked_sub(1).is_some_and(|prior| adjacent(&items[prior], &items[index]))
                || items.get(index + 1).is_some_and(|next| adjacent(&items[index], next))
        })
        .collect()
}

fn regex_error(error: regex::Error) -> Error {
    Error::Regex { pattern: "list".into(), message: error.to_string() }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn alphabetical_sequences_are_protected_and_separated() {
        assert_eq!(
            add_line_breaks("a. The first item. b. The second item. c. The third list item", ListMode::Standard).unwrap(),
            "a∯ The first item.\rb∯ The second item.\rc∯ The third list item"
        );
    }
}
