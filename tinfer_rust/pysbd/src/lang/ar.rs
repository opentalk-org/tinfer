use crate::{Result, Rule, Rules};

const ABBREVIATIONS: &[&str] = &[
    "ا", "ا. د", "ا.د", "ا.ش.ا", "ا.ش.ا", "إلخ", "ت.ب", "ت.ب",
    "ج.ب", "جم", "ج.ب", "ج.م.ع", "ج.م.ع", "س.ت", "س.ت", "سم",
    "ص.ب.", "ص.ب", "كج.", "كلم.", "م", "م.ب", "م.ب", "ه",
];
const PREPOSITIVE: &[&str] = &[
];
const NUMBER: &[&str] = &[
];

pub(crate) const fn rules() -> Rules {
    let mut rules = Rules::standard();
    rules.sentence_boundary = r".*?[:\.!\?؟،]|.*?\z|.*?$";
    rules.punctuations = &['?', '!', ':', '.', '؟', '،'];
    rules.abbreviations = ABBREVIATIONS;
    rules.prepositive_abbreviations = PREPOSITIVE;
    rules.number_abbreviations = NUMBER;
    rules.protect_all_abbreviation_periods = true;
    rules
}

pub(crate) fn before_boundaries(text: &str) -> Result<String> {
    let output = Rule::new(r"(?<=\d):(?=\d)", "♭").replace_all(text)?;
    let output = Rule::new(r"،(?=\s\S+،)", "♬").replace_all(&output)?;
    Rule::new(r"&ᓴ&$", "!").replace_all(&output)
}
