use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};

use fancy_regex::Regex as FancyRegex;
use regex::Regex;

use crate::{Error, Result};

#[derive(Clone)]
enum CompiledRegex {
    Linear(Regex),
    Fancy(FancyRegex),
}

static REGEX_CACHE: OnceLock<Mutex<HashMap<String, CompiledRegex>>> = OnceLock::new();

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Rule {
    pub pattern: String,
    pub replacement: String,
}

impl Rule {
    pub fn new(pattern: impl Into<String>, replacement: impl Into<String>) -> Self {
        Self { pattern: pattern.into(), replacement: replacement.into() }
    }

    pub fn replace_all(&self, text: &str) -> Result<String> {
        let replacement = python_replacement(&self.replacement);
        match compiled_regex(&self.pattern)? {
            CompiledRegex::Linear(regex) => Ok(regex.replace_all(text, replacement.as_str()).into_owned()),
            CompiledRegex::Fancy(regex) => regex
                .try_replacen(text, 0, replacement.as_str())
                .map(|output| output.into_owned())
                .map_err(|error| Error::Regex { pattern: self.pattern.clone(), message: error.to_string() }),
        }
    }
}

fn compiled_regex(pattern: &str) -> Result<CompiledRegex> {
    let cache = REGEX_CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    let cached = {
        let cache = cache.lock().expect("regex cache mutex poisoned");
        cache.get(pattern).cloned()
    };
    if let Some(regex) = cached {
        return Ok(regex);
    }
    let compiled = match Regex::new(pattern) {
        Ok(regex) => CompiledRegex::Linear(regex),
        Err(linear_error) => CompiledRegex::Fancy(
            FancyRegex::new(pattern)
                .map_err(|fancy_error| Error::Regex { pattern: pattern.into(), message: format!("{linear_error}; {fancy_error}") })?,
        ),
    };
    let mut cache = cache.lock().expect("regex cache mutex poisoned");
    Ok(cache.entry(pattern.to_owned()).or_insert(compiled).clone())
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Rules {
    pub sentence_boundary: &'static str,
    pub punctuations: &'static [char],
    pub multi_period_abbreviation: &'static str,
    pub abbreviations: &'static [&'static str],
    pub prepositive_abbreviations: &'static [&'static str],
    pub number_abbreviations: &'static [&'static str],
    pub sentence_starters: &'static [&'static str],
    pub(crate) abbreviation_mode: AbbreviationMode,
    pub(crate) list_mode: ListMode,
    pub(crate) number_mode: NumberMode,
    pub(crate) punctuation_mode: PunctuationMode,
    pub(crate) boundary_mode: BoundaryMode,
    pub(crate) date_words: &'static [&'static str],
    pub abbreviation_rules: &'static [(&'static str, &'static str)],
    pub abbreviation_rules_only: bool,
    pub abbreviations_before_uppercase: bool,
    pub protect_all_abbreviation_periods: bool,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum AbbreviationMode {
    Standard,
    Persian,
    Russian,
    Slovak,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum ListMode {
    Standard,
    NoAlphabetical,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum NumberMode {
    Standard,
    Slovak,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum PunctuationMode {
    Standard,
    Slovak,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum BoundaryMode {
    Standard,
    Persian,
}

impl Rules {
    pub const fn standard() -> Self {
        Self {
            sentence_boundary: r#"（(?:[^）])*）(?=\s?[A-Z])|「(?:[^」])*」(?=\s[A-Z])|\((?:[^\)]){2,}\)(?=\s[A-Z])|'(?:[^'])*[^,]'(?=\s[A-Z])|"(?:[^"])*[^,]"(?=\s[A-Z])|“(?:[^”])*[^,]”(?=\s[A-Z])|[。．.！!?？ ]{2,}|\S.*?[。．.！!?？ȸȹ☉☈☇☄]|[。．.！!?？]"#,
            punctuations: &['。', '．', '.', '！', '!', '?', '？'],
            multi_period_abbreviation: r"\b[a-z](?:\.[a-z])+[.]",
            abbreviations: &[],
            prepositive_abbreviations: &[],
            number_abbreviations: &[],
            sentence_starters: &[],
            abbreviation_mode: AbbreviationMode::Standard,
            list_mode: ListMode::Standard,
            number_mode: NumberMode::Standard,
            punctuation_mode: PunctuationMode::Standard,
            boundary_mode: BoundaryMode::Standard,
            date_words: &[],
            abbreviation_rules: &[],
            abbreviation_rules_only: false,
            abbreviations_before_uppercase: false,
            protect_all_abbreviation_periods: false,
        }
    }
}

pub fn apply_rules(text: &str, rules: &[Rule]) -> Result<String> {
    let mut output = text.to_owned();
    for rule in rules {
        output = rule.replace_all(&output)?;
    }
    Ok(output)
}

pub fn python_replacement(replacement: &str) -> String {
    let mut converted = String::with_capacity(replacement.len());
    let mut chars = replacement.chars().peekable();
    while let Some(character) = chars.next() {
        match (character, chars.peek().copied()) {
            ('$', _) => converted.push_str("$$"),
            ('\\', Some('\\')) => {
                chars.next();
                converted.push('\\');
            }
            ('\\', Some('n')) => {
                chars.next();
                converted.push('\n');
            }
            ('\\', Some('r')) => {
                chars.next();
                converted.push('\r');
            }
            ('\\', Some('t')) => {
                chars.next();
                converted.push('\t');
            }
            ('\\', Some('g')) => convert_named_group(&mut chars, &mut converted),
            ('\\', Some(next)) if next.is_ascii_digit() => {
                converted.push('$');
                converted.push(chars.next().expect("peeked replacement digit"));
            }
            _ => converted.push(character),
        }
    }
    converted
}

fn convert_named_group<I>(chars: &mut std::iter::Peekable<I>, converted: &mut String)
where
    I: Iterator<Item = char>,
{
    chars.next();
    assert_eq!(chars.next(), Some('<'), "invalid Python replacement group");
    converted.push_str("${");
    loop {
        let character = chars.next().expect("unterminated Python replacement group");
        if character == '>' {
            converted.push('}');
            break;
        }
        converted.push(character);
    }
}
