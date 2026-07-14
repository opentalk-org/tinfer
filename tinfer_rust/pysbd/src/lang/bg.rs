use crate::Rules;

const ABBREVIATIONS: &[&str] = &[
    "p.s", "акад", "ал", "б.р", "б.ред", "бел.а", "бел.пр", "бр",
    "бул", "в", "вж", "вкл", "вм", "вр", "г", "ген",
    "гр", "дж", "дм", "доц", "др", "ем", "заб", "зам",
    "инж", "к.с", "кв", "кв.м", "кг", "км", "кор", "куб",
    "куб.м", "л", "лв", "м", "м.г", "мин", "млн", "млрд",
    "мм", "н.с", "напр", "пл", "полк", "проф", "р", "рис",
    "с", "св", "сек", "см", "сп", "срв", "ст", "стр",
    "т", "т.г", "т.е", "т.н", "т.нар", "табл", "тел", "у",
    "ул", "фиг", "ха", "хил", "ч", "чл", "щ.д",
];
const PREPOSITIVE: &[&str] = &[
];
const NUMBER: &[&str] = &[
];

pub(crate) const fn rules() -> Rules {
    let mut rules = Rules::standard();
    rules.abbreviations = ABBREVIATIONS;
    rules.prepositive_abbreviations = PREPOSITIVE;
    rules.number_abbreviations = NUMBER;
    rules.sentence_starters = &[];
    rules.protect_all_abbreviation_periods = true;
    rules
}
