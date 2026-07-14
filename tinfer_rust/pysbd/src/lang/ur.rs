use crate::{Rules, lang::english_rules};

const SENTENCE_STARTERS: &[&str] = &[];

pub(crate) const fn rules() -> Rules {
    let mut rules = english_rules();
    rules.sentence_starters = SENTENCE_STARTERS;
    rules.sentence_boundary = r#".*?[۔؟!\?]|.*?$"#;
    rules.punctuations = &['?', '!', '۔', '؟'];
    rules
}
