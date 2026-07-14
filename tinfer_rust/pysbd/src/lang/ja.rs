use crate::{Result, Rule, Rules};

const SENTENCE_STARTERS: &[&str] = &[];
const NEW_LINE_IN_MIDDLE_OF_WORD_RULE: (&str, &str) = (r"(?<=の)\n(?=\S)", "");
const BETWEEN_PARENS_JA_REGEX: &str = r"（[^（）]*）";
const BETWEEN_QUOTE_JA_REGEX: &str = r"「[^「」]*」";

pub(crate) const fn rules() -> Rules {
    let mut rules = Rules::standard();
    rules.sentence_starters = SENTENCE_STARTERS;
    rules
}

pub(crate) fn clean(text: &str) -> Result<String> {
    Rule::new(NEW_LINE_IN_MIDDLE_OF_WORD_RULE.0, NEW_LINE_IN_MIDDLE_OF_WORD_RULE.1).replace_all(text)
}

pub(crate) fn between_punctuation(text: &str) -> Result<String> {
    let output = crate::punctuation::protect_pattern(text, BETWEEN_PARENS_JA_REGEX, true)?;
    crate::punctuation::protect_pattern(&output, BETWEEN_QUOTE_JA_REGEX, true)
}
