use fancy_regex::{Captures, Regex};

use crate::{Error, PunctuationMode, Result};

const BETWEEN: &[&str] = &[
    r"(?<=\s)'(?:[^']|'[a-zA-Z])*'",
    r"(?<=\s)‘(?:[^’]|’[a-zA-Z])*’",
    r#""(?=(?P<tmp>[^"\\]+|\\{2}|\\.)*)(?P=tmp)""#,
    r"\[(?=(?P<tmp>[^\]\\]+|\\{2}|\\.)*)(?P=tmp)\]",
    r"\((?=(?P<tmp>[^\(\)\\]+|\\{2}|\\.)*)(?P=tmp)\)",
    r"«(?=(?P<tmp>[^»\\]+|\\{2}|\\.)*)(?P=tmp)»",
    r"--(?=(?P<tmp>[^--]*))(?P=tmp)--",
    r"“(?=(?P<tmp>[^”\\]+|\\{2}|\\.)*)(?P=tmp)”",
];

pub(crate) fn between(text: &str, mode: PunctuationMode) -> Result<String> {
    between_patterns(text, false, mode)
}

pub(crate) fn between_without_slanted_quotes(text: &str) -> Result<String> {
    between_patterns(text, true, PunctuationMode::Standard)
}

fn between_patterns(text: &str, skip_slanted_quotes: bool, mode: PunctuationMode) -> Result<String> {
    let mut output = text.to_owned();
    let parentheses =
        regex::Regex::new(r"\([^()]*\)").map_err(|error| Error::Regex { pattern: BETWEEN[4].into(), message: error.to_string() })?;
    for (index, pattern) in BETWEEN.iter().enumerate() {
        if index == 7 && (skip_slanted_quotes || !output.contains('”')) {
            continue;
        }
        if index == 0 && leading_apostrophe_word(&output)? && !output.contains("' ") {
            continue;
        }
        if index == 4 {
            output = parentheses.replace_all(&output, |captures: &regex::Captures<'_>| protect(&captures[0], true)).into_owned();
        } else {
            let regex = Regex::new(pattern).map_err(|error| compile_error(pattern, error))?;
            output = regex
                .try_replacen(&output, 0, |captures: &Captures<'_>| protect(&captures[0], index != 0))
                .map_err(|error| runtime_error(pattern, error))?
                .into_owned();
        }
    }
    if mode == PunctuationMode::Slovak {
        let pattern = r#"„(?=(?P<tmp>[^“\\]+|\\{2}|\\.)*)(?P=tmp)“"#;
        let regex = Regex::new(pattern).map_err(|error| compile_error(pattern, error))?;
        output = regex
            .try_replacen(&output, 0, |captures: &Captures<'_>| protect(&captures[0], true))
            .map_err(|error| runtime_error(pattern, error))?
            .into_owned();
    }
    Ok(output)
}

pub(crate) fn protect(text: &str, quotes: bool) -> String {
    let mut output = text.replace('.', "∯").replace('。', "&ᓰ&").replace('．', "&ᓱ&");
    output = output.replace('！', "&ᓳ&").replace('!', "&ᓴ&").replace('?', "&ᓷ&").replace('？', "&ᓸ&");
    if quotes {
        output = output.replace('\'', "&⎋&");
    }
    output
}

pub(crate) fn protect_pattern(text: &str, pattern: &str, quotes: bool) -> Result<String> {
    let regex = regex::Regex::new(pattern).map_err(|error| Error::Regex { pattern: pattern.into(), message: error.to_string() })?;
    Ok(regex.replace_all(text, |captures: &regex::Captures<'_>| protect(&captures[0], quotes)).into_owned())
}

fn leading_apostrophe_word(text: &str) -> Result<bool> {
    Regex::new(r"(?<=\s)'(?:[^']|'[a-zA-Z])*'\S")
        .map_err(|error| compile_error("apostrophe", error))?
        .is_match(text)
        .map_err(|error| runtime_error("apostrophe", error))
}

fn compile_error(pattern: &str, error: fancy_regex::Error) -> Error {
    Error::Regex { pattern: pattern.into(), message: error.to_string() }
}

fn runtime_error(pattern: &str, error: fancy_regex::Error) -> Error {
    Error::Regex { pattern: pattern.into(), message: error.to_string() }
}
