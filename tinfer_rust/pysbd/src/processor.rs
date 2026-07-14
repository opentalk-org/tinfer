use fancy_regex::{Captures, Regex};

use crate::{Error, Language, NumberMode, Result, Rule, Rules, abbreviation, lang, lists, punctuation};

const EXCLAMATION_WORDS: &[&str] = &[
    "!Xũ",
    "!Kung",
    "ǃʼOǃKung",
    "!Xuun",
    "!Kung-Ekoka",
    "ǃHu",
    "ǃKhung",
    "ǃKu",
    "ǃung",
    "ǃXo",
    "ǃXû",
    "ǃXung",
    "ǃXũ",
    "!Xun",
    "Yahoo!",
    "Y!J",
    "Yum!",
];

pub(crate) fn process(text: &str, language: Language) -> Result<Vec<String>> {
    if text.is_empty() {
        return Ok(Vec::new());
    }
    let rules = language.rules();
    let mut prepared = lists::add_line_breaks(&text.replace('\n', "\r"), rules.list_mode)?;
    prepared = abbreviation::replace(&prepared, &rules, language)?;
    prepared = replace_numbers(&prepared)?;
    prepared = lang::replace_numbers(&prepared, language)?;
    if rules.number_mode == NumberMode::Slovak {
        prepared = replace_slovak_numbers(&prepared, rules.date_words)?;
    }
    prepared = replace_continuous_punctuation(&prepared)?;
    prepared = Rule::new(r"(?<=[^\d\s])(\.|∯)((\[(\d{1,3},?\s?-?\s?)*\b\d{1,3}\])+|((\d{1,3}\s?)?\d{1,3}))(\s)(?=[A-Z])", r"∯\2\r\7")
        .replace_all(&prepared)?;
    prepared = Rule::new(r"([a-zA-Z0-9_])(\.)([a-zA-Z0-9_])", r"\1∮\3").replace_all(&prepared)?;
    prepared = Rule::new(r"(?<=[a-zA-Z]°)\.(?=\s*\d+)", "∯").replace_all(&prepared)?;
    prepared = Rule::new(
        r"(?<=\s)\.(?=(jpe?g|png|gif|tiff?|pdf|ps|docx?|xlsx?|svg|bmp|tga|exif|odt|html?|txt|rtf|bat|sxw|xml|zip|exe|msi|blend|wmv|mp[34]|pptx?|flac|rb|cpp|cs|js)\s)",
        "∯",
    ).replace_all(&prepared)?;
    split_into_segments(&prepared, &rules, language)
}

fn split_into_segments(text: &str, rules: &Rules, language: Language) -> Result<Vec<String>> {
    let prepared = protect_parentheses_between_quotes(text)?;
    let mut raw = Vec::new();
    for part in prepared.split('\r').filter(|part| !part.is_empty()) {
        let part = apply_ellipsis(&Rule::new(r"\n", "ȹ").replace_all(part)?)?;
        if rules.punctuations.iter().any(|punctuation| part.contains(*punctuation)) {
            raw.extend(process_text(&part, rules, language)?);
        } else {
            raw.push(part);
        }
    }
    let mut output = Vec::new();
    for sentence in raw {
        let restored = restore_symbols(&sentence)?;
        output.extend(post_process(&restored)?.into_iter().filter(|part| !part.is_empty()));
    }
    output.into_iter().map(|sentence| Rule::new(r"&⎋&", "'").replace_all(&sentence)).collect()
}

fn process_text(text: &str, rules: &Rules, language: Language) -> Result<Vec<String>> {
    let mut prepared = text.to_owned();
    if !prepared.ends_with(rules.punctuations) {
        prepared.push('ȸ');
    }
    for word in EXCLAMATION_WORDS {
        prepared = prepared.replace(word, &punctuation::protect(word, true));
    }
    prepared = lang::between_punctuation(&prepared, language, rules.punctuation_mode)?;
    let double = Regex::new(r"\?!|!\?|\?\?|!!").map_err(regex_error)?.find(&prepared).map_err(runtime_error)?;
    if double.is_none_or(|found| found.start() != 0) {
        for (pattern, replacement) in [(r"\?!", "☉"), (r"!\?", "☈"), (r"\?\?", "☇"), (r"!!", "☄")] {
            prepared = Rule::new(pattern, replacement).replace_all(&prepared)?;
        }
    }
    for (pattern, replacement) in [(r#"\?(?=('|"))"#, "&ᓷ&"), (r#"!(?=('|"))"#, "&ᓴ&"), (r"!(?=,\s[a-z])", "&ᓴ&"), (r"!(?=\s[a-z])", "&ᓴ&")]
    {
        prepared = Rule::new(pattern, replacement).replace_all(&prepared)?;
    }
    prepared = lists::protect_roman_parens(&prepared)?;
    prepared = lang::before_boundaries(&prepared, language, rules.boundary_mode)?;
    sentence_boundaries(&prepared, rules.sentence_boundary)
}

fn replace_slovak_numbers(text: &str, months: &[&str]) -> Result<String> {
    let mut output = text.to_owned();
    for month in months {
        let pattern = format!(r"(?<=\d)\.(?=\s*{})", regex::escape(month));
        output = Rule::new(pattern, "∯").replace_all(&output)?;
    }
    output = Rule::new(r"(?<=\d)\.(?=\s*[a-z]+)", "∯").replace_all(&output)?;
    Rule::new(r"(?i)((\s+[VXI]+)|(^[VXI]+))(\.)(?=\s+)", r"\1∯").replace_all(&output)
}

fn replace_numbers(text: &str) -> Result<String> {
    let mut output = text.to_owned();
    for pattern in [r"\.(?=\d)", r"(?<=\d)\.(?=\S)", r"(?<=\r\d)\.(?=(\s\S)|\))", r"(?<=^\d)\.(?=(\s\S)|\))", r"(?<=^\d\d)\.(?=(\s\S)|\))"]
    {
        output = Rule::new(pattern, "∯").replace_all(&output)?;
    }
    Ok(output)
}

fn replace_continuous_punctuation(text: &str) -> Result<String> {
    let pattern = r"(?<=\S)(!|\?){3,}(?=(\s|\z|$))";
    let regex = Regex::new(pattern).map_err(regex_error)?;
    regex
        .try_replacen(text, 0, |captures: &Captures<'_>| captures[0].replace('!', "&ᓴ&").replace('?', "&ᓷ&"))
        .map(|value| value.into_owned())
        .map_err(runtime_error)
}

fn protect_parentheses_between_quotes(text: &str) -> Result<String> {
    let pattern = r#"["”]\s\(.*\)\s["“]"#;
    let regex = Regex::new(pattern).map_err(regex_error)?;
    regex
        .try_replacen(text, 0, |captures: &Captures<'_>| {
            let value = Rule::new(r"\s(?=\()", "\r").replace_all(&captures[0]).expect("static regex");
            Rule::new(r"(?<=\))\s", "\r").replace_all(&value).expect("static regex")
        })
        .map(|value| value.into_owned())
        .map_err(runtime_error)
}

fn sentence_boundaries(text: &str, pattern: &str) -> Result<Vec<String>> {
    let regex = Regex::new(pattern).map_err(regex_error)?;
    let mut output = Vec::new();
    for result in regex.find_iter(text) {
        let found = result.map_err(runtime_error)?;
        if !found.as_str().is_empty() {
            output.push(found.as_str().to_owned());
        }
    }
    Ok(output)
}

fn apply_ellipsis(text: &str) -> Result<String> {
    let mut output = text.to_owned();
    for (pattern, replacement) in [
        (r"(\s\.){3}\s", "♟♟♟♟♟♟♟"),
        (r"(?<=[a-z])(\.\s){3}\.($|\\n)", "♝♝♝♝♝♝♝"),
        (r"(?<=\S)\.{3}(?=\.\s[A-Z])", "ƪƪƪ"),
        (r"\.\.\.(?=\s+[A-Z])", "☏☏."),
        (r"\.\.\.", "ƪƪƪ"),
    ] {
        output = Rule::new(pattern, replacement).replace_all(&output)?;
    }
    Ok(output)
}

fn post_process(text: &str) -> Result<Vec<String>> {
    if text.len() > 2 && text.chars().all(|character| character.is_ascii_alphabetic()) {
        return Ok(vec![text.to_owned()]);
    }
    let mut output = text.to_owned();
    for (pattern, replacement) in [(r"ƪƪƪ", "..."), (r"♟♟♟♟♟♟♟", " . . . "), (r"♝♝♝♝♝♝♝", ". . . ."), (r"☏☏", ".."), (r"∮", ".")]
    {
        output = Rule::new(pattern, replacement).replace_all(&output)?;
    }
    let quote_end = Regex::new(r#"[!?\.-]["'“”]\s[A-Z]"#).map_err(regex_error)?.is_match(&output).map_err(runtime_error)?;
    if quote_end {
        let split = Regex::new(r#"(?<=[!?\.-]["'“”])\s(?=[A-Z])"#).map_err(regex_error)?;
        return split.split(&output).map(|part| part.map(str::to_owned).map_err(runtime_error)).collect();
    }
    Ok(vec![output.replace('\n', "").trim().to_owned()])
}

fn restore_symbols(text: &str) -> Result<String> {
    let mut output = text.to_owned();
    for (pattern, replacement) in [
        ("∯", "."),
        ("♬", "،"),
        ("♭", ":"),
        ("&ᓰ&", "。"),
        ("&ᓱ&", "．"),
        ("&ᓳ&", "！"),
        ("&ᓴ&", "!"),
        ("&ᓷ&", "?"),
        ("&ᓸ&", "？"),
        ("☉", "?!"),
        ("☇", "??"),
        ("☈", "!?"),
        ("☄", "!!"),
        ("&✂&", "("),
        ("&⌬&", ")"),
        ("ȸ", ""),
        ("ȹ", "\n"),
    ] {
        output = Rule::new(pattern, replacement).replace_all(&output)?;
    }
    Ok(output)
}

fn regex_error(error: fancy_regex::Error) -> Error {
    Error::Regex { pattern: "processor".into(), message: error.to_string() }
}

fn runtime_error(error: fancy_regex::Error) -> Error {
    Error::Regex { pattern: "processor".into(), message: error.to_string() }
}
