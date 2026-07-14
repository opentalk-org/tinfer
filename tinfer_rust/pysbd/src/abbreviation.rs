use fancy_regex::{Captures, Regex};

use crate::{AbbreviationMode, Error, Language, Result, Rule, Rules, lang};

pub(crate) fn replace(text: &str, rules: &Rules, language: Language) -> Result<String> {
    let mut output = Rule::new(r"\.(?='s\s)|\.(?='s$)|\.(?='s\z)", "∯").replace_all(text)?;
    output = Rule::new(r"(?<=Co)\.(?=\sKG)", "∯").replace_all(&output)?;
    output = Rule::new(r"(?<=^[A-Z])\.(?=\s)", "∯").replace_all(&output)?;
    output = Rule::new(r"(?<=\s[A-Z])\.(?=,?\s)", "∯").replace_all(&output)?;
    for (pattern, replacement) in rules.abbreviation_rules {
        output = Rule::new(*pattern, *replacement).replace_all(&output)?;
    }
    if rules.abbreviation_rules_only {
        return replace_multi_periods(&output, rules.multi_period_abbreviation);
    }
    let mut handled = String::with_capacity(output.len());
    for line in output.split_inclusive(['\n', '\r']) {
        handled.push_str(&replace_line(line, rules)?);
    }
    if handled.is_empty() {
        handled = replace_line(&output, rules)?;
    }
    output = replace_multi_periods(&handled, rules.multi_period_abbreviation)?;
    for (pattern, replacement) in
        [(r"(?<= P∯M)∯(?=\s[A-Z])", "."), (r"(?<=A∯M)∯(?=\s[A-Z])", "."), (r"(?<=p∯m)∯(?=\s[A-Z])", "."), (r"(?<=a∯m)∯(?=\s[A-Z])", ".")]
    {
        output = Rule::new(pattern, replacement).replace_all(&output)?;
    }
    lang::restore_abbreviation_boundaries(&output, language, rules.sentence_starters)
}

fn replace_line(text: &str, rules: &Rules) -> Result<String> {
    let mut output = text.to_owned();
    let lowered = text.to_lowercase();
    for abbreviation in rules.abbreviations {
        if !lowered.contains(abbreviation.trim()) {
            continue;
        }
        let escaped = regex::escape(abbreviation.trim()).replace(r"\.", "[.∯]");
        if rules.protect_all_abbreviation_periods {
            let placeholder_aware = escaped.replace(r"\.", "[.∯]");
            let pattern = format!(r"(?i)(^|\s)({placeholder_aware})\.");
            let regex = regex::Regex::new(&pattern).map_err(|error| Error::Regex { pattern, message: error.to_string() })?;
            output = regex.replace_all(&output, |captures: &regex::Captures<'_>| captures[0].replace('.', "∯")).into_owned();
            continue;
        }
        if rules.abbreviation_mode == AbbreviationMode::Slovak
            && !rules.prepositive_abbreviations.contains(&abbreviation.trim())
            && !rules.number_abbreviations.contains(&abbreviation.trim())
        {
            let replacement = format!("{}∯", abbreviation.replace('.', "∯"));
            output = output.replace(&format!("{abbreviation}."), &replacement);
            continue;
        }
        let pattern = if rules.abbreviation_mode == AbbreviationMode::Persian {
            format!(r"(?<=(?i:\s{escaped}))\.")
        } else if rules.abbreviations_before_uppercase {
            let protected = escaped.replace(r"\.", "[.∯]");
            format!(r"(?<=(?i:\s{protected}))\.(?=\s)")
        } else if rules.prepositive_abbreviations.contains(&abbreviation.trim()) {
            format!(r"(?<=(?i:\s{escaped}))\.(?=(\s|:\d+))")
        } else if rules.number_abbreviations.contains(&abbreviation.trim()) {
            format!(r"(?<=(?i:\s{escaped}))\.(?=(\s\d|\s+\())")
        } else if rules.abbreviation_mode == AbbreviationMode::Russian {
            format!(r"(?<=(?i:\s{escaped}))\.")
        } else {
            format!(r"(?<=(?i:\s{escaped}))\.(?=[.:\-?,]|\s(?:[a-z]|\d|\(|I(?:\s|'m|'ll)))")
        };
        output.insert(0, ' ');
        output = Rule::new(pattern, "∯").replace_all(&output)?;
        output.remove(0);
    }
    Ok(output)
}

fn replace_multi_periods(text: &str, pattern: &str) -> Result<String> {
    let regex = Regex::new(&format!("(?i){pattern}")).map_err(|error| compile_error(pattern, error))?;
    regex
        .try_replacen(text, 0, |captures: &Captures<'_>| captures[0].replace('.', "∯"))
        .map(|value| value.into_owned())
        .map_err(|error| runtime_error(pattern, error))
}

fn compile_error(pattern: &str, error: fancy_regex::Error) -> Error {
    Error::Regex { pattern: pattern.into(), message: error.to_string() }
}

fn runtime_error(pattern: &str, error: fancy_regex::Error) -> Error {
    Error::Regex { pattern: pattern.into(), message: error.to_string() }
}
