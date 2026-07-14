use crate::{AbbreviationMode, Rules};

#[rustfmt::skip]
const ABBREVIATIONS: &[&str] = &[
    "y", "y.e", "а", "авт", "адм.-терр", "акад",
    "в", "вв", "вкз", "вост.-европ", "г", "гг",
    "гос", "гр", "д", "деп", "дисс", "дол",
    "долл", "ежедн", "ж", "жен", "з", "зап",
    "зап.-европ", "заруб", "и", "ин", "иностр", "инст",
    "к", "канд", "кв", "кг", "куб", "л",
    "л.h", "л.н", "м", "мин", "моск", "муж",
    "н", "нед", "о", "п", "пгт", "пер",
    "пп", "пр", "просп", "проф", "р", "руб",
    "с", "сек", "см", "спб", "стр", "т",
    "тел", "тов", "тт", "тыс", "у", "у.е",
    "ул", "ф", "ч",
];
const PREPOSITIVE_ABBREVIATIONS: &[&str] = &[];
const NUMBER_ABBREVIATIONS: &[&str] = &[];
const SENTENCE_STARTERS: &[&str] = &[];

pub(crate) const fn rules() -> Rules {
    let mut rules = Rules::standard();
    rules.abbreviations = ABBREVIATIONS;
    rules.prepositive_abbreviations = PREPOSITIVE_ABBREVIATIONS;
    rules.number_abbreviations = NUMBER_ABBREVIATIONS;
    rules.sentence_starters = SENTENCE_STARTERS;
    rules.abbreviation_mode = AbbreviationMode::Russian;
    rules
}
