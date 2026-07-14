use fancy_regex::Regex;

use crate::{DocType, Language, Result, Rule, apply_rules};

const HTML: &[(&str, &str)] = &[(r#"</?\w+((\s+\w+(\s*=\s*(?:\".*?\"|'.*?'|[\^'\">\s]+))?)+\s*|\s*)/?>"#, ""), (r"&lt;/?[^gt;]*gt;", "")];

pub fn clean(text: &str, language: Language, doc_type: Option<DocType>) -> Result<String> {
    if text.is_empty() {
        return Ok(String::new());
    }
    let prepared = crate::lang::clean(text, language)?;
    let mut output = replace(&prepared, r"(?<=\s)\n(?=([a-z]|\())", "")?;
    output = replace(&output, r"\n(?=[a-zA-Z]{1,2}\n)", "")?;
    output = replace(&output, r"\n \n", "\r")?;
    output = replace(&output, r"\n\n", "\r")?;
    output = match doc_type {
        Some(DocType::Pdf) => clean_pdf(&output)?,
        None => replace(&replace(&output, r"\n(?=\.(\s|\n))", "")?, r"\n", "\r")?,
    };
    for (pattern, replacement) in [(r"\\n", "\n"), (r"\\r", "\r"), (r"\\\ n", "\n"), (r"\\\ r", "\r")] {
        output = replace(&output, pattern, replacement)?;
    }
    let html_rules: Vec<Rule> = HTML.iter().map(|(pattern, replacement)| Rule::new(*pattern, *replacement)).collect();
    output = apply_rules(&output, &html_rules)?;
    output = protect_bracket_questions(&output)?;
    output = replace(&output, r"{b\^&gt;\d*&lt;b\^}|{b\^>\d*<b\^}", "")?;
    output = output.replace('`', "'");
    output = replace(&replace(&output, "''", "\"")?, "``", "\"")?;
    output = replace(&output, r"\.{4,}\s*\d+-*\d*", "\r")?;
    output = replace(&output, r"\.{5,}", " ")?;
    output = replace(&output, r"/{3}", "")?;
    output = separate_connected_sentences(&output)?;
    output = replace(&output, r"\.{5,}", " ")?;
    replace(&output, r"/{3}", "")
}

fn clean_pdf(text: &str) -> Result<String> {
    let output = replace(text, r"\n(?=•')", "\r")?;
    let output = replace(&output, r"(?<=[^\n]\s)\n(?=\S)", "")?;
    replace(&output, r"\n(?=[a-z])", " ")
}

fn protect_bracket_questions(text: &str) -> Result<String> {
    let regex = Regex::new(r"\[(?:[^\]])*\]").map_err(regex_error)?;
    regex
        .try_replacen(text, 0, |captures: &fancy_regex::Captures<'_>| captures[0].replace('?', "&ᓷ&"))
        .map(|value| value.into_owned())
        .map_err(runtime_error)
}

fn separate_connected_sentences(text: &str) -> Result<String> {
    let mut output = text.to_owned();
    for word in text.split(' ') {
        if word.contains(['@']) || ["http", ".com", "net", "www", "//"].iter().any(|part| word.contains(part)) {
            continue;
        }
        if Regex::new(r"(?<=[a-z])\.(?=[A-Z])").map_err(regex_error)?.is_match(word).map_err(runtime_error)? {
            output = output.replace(word, &replace(word, r"(?<=[a-z])\.(?=[A-Z])", ". ")?);
        }
        if Regex::new(r"(?<=\d)\.(?=[A-Z])").map_err(regex_error)?.is_match(word).map_err(runtime_error)? {
            output = output.replace(word, &replace(word, r"(?<=\d)\.(?=[A-Z])", ". ")?);
        }
    }
    Ok(output)
}

fn replace(text: &str, pattern: &str, replacement: &str) -> Result<String> {
    Rule::new(pattern, replacement).replace_all(text)
}

fn regex_error(error: fancy_regex::Error) -> crate::Error {
    crate::Error::Regex { pattern: "cleaner".into(), message: error.to_string() }
}

fn runtime_error(error: fancy_regex::Error) -> crate::Error {
    crate::Error::Regex { pattern: "cleaner".into(), message: error.to_string() }
}
