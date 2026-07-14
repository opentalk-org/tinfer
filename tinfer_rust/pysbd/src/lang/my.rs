use crate::Rules;

pub(crate) const fn rules() -> Rules {
    let mut rules = Rules::standard();
    rules.sentence_boundary = r".*?[။၏!\?]|.*?$";
    rules.punctuations = &['။', '၏', '?', '!'];
    rules
}
