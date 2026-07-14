use crate::Rules;

const SENTENCE_BOUNDARY_REGEX: &str = r".*?[\.;!\?]|.*?$";
const PUNCTUATIONS: &[char] = &['.', '!', ';', '?'];
const SENTENCE_STARTERS: &[&str] = &[];

pub(crate) const fn rules() -> Rules {
    let mut rules = Rules::standard();
    rules.sentence_boundary = SENTENCE_BOUNDARY_REGEX;
    rules.punctuations = PUNCTUATIONS;
    rules.sentence_starters = SENTENCE_STARTERS;
    rules
}
