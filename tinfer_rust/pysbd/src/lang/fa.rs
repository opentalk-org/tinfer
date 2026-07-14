use crate::{AbbreviationMode, BoundaryMode, Rules, lang::english_rules};

const SENTENCE_STARTERS: &[&str] = &[];

pub(crate) const fn rules() -> Rules {
    let mut rules = english_rules();
    rules.sentence_starters = SENTENCE_STARTERS;
    rules.sentence_boundary = r#".*?[:\.!\?؟]|.*?\Z|.*?$"#;
    rules.punctuations = &['?', '!', ':', '.', '؟'];
    rules.abbreviation_mode = AbbreviationMode::Persian;
    rules.boundary_mode = BoundaryMode::Persian;
    rules
}
