use crate::{AbbreviationMode, ListMode, NumberMode, PunctuationMode, Rules};

#[rustfmt::skip]
const ABBREVIATIONS: &[&str] = &[
    "č", "no", "nr", "s. r. o", "ing", "p",
    "a. d", "o. k", "pol. pr", "a. s. a. p", "p. n. l", "red",
    "o.k", "a.d", "m.o", "pol.pr", "a.s.a.p", "p.n.l",
    "pp", "sl", "corp", "plgr", "tz", "rtg",
    "o.c.p", "o. c. p", "c.k", "c. k", "n.a", "n. a",
    "a.m", "a. m", "vz", "i.b", "i. b", "ú.p.v.o",
    "ú. p. v. o", "bros", "rsdr", "doc", "tu", "ods",
    "n.w.a", "n. w. a", "nár", "pedg", "paeddr", "rndr",
    "naprk", "a.g.p", "a. g. p", "prof", "pr", "a.v",
    "a. v", "por", "mvdr", "nešp", "u.s", "u. s",
    "kt", "vyd", "e.t", "e. t", "al", "ll.m",
    "ll. m", "o.f.i", "o. f. i", "mr", "apod", "súkr",
    "stred", "s.e.g", "s. e. g", "sr", "tvz", "ind",
    "var", "etc", "atd", "n.o", "n. o", "s.a",
    "s. a", "např", "a.i.i", "a. i. i", "a.k.a", "a. k. a",
    "konkr", "čsl", "odd", "ltd", "t.z", "t. z",
    "o.z", "o. z", "obv", "obr", "pok", "tel",
    "št", "skr", "phdr", "xx", "š.p", "š. p",
    "ph.d", "ph. d", "m.n.m", "m. n. m", "zz", "roz",
    "atď.", "ev", "v.sp", "v. sp", "drsc", "mudr",
    "t.č", "t. č", "el", "os", "co", "r.o",
    "r. o", "str", "p.a", "p. a", "zdravot", "prek",
    "gen", "viď", "dr", "cca", "p.s", "p. s",
    "zák", "slov", "arm", "inc", "max", "d.c",
    "k.o", "a. r. k", "d. c", "k. o", "a. r. k", "soc",
    "bc", "zs", "akad", "sz", "pozn", "tr",
    "nám", "kol", "csc", "ul", "sp", "o.i",
    "jr", "zb", "sv", "tj", "čs", "tzn",
    "príp", "iv", "hl", "st", "pod", "vi",
    "tis", "stor", "rozh", "mld", "atď", "mgr",
    "a.s", "a. s", "phd", "z.z", "z. z", "judr",
    "ing", "hod", "vs", "písm", "s.r.o", "min",
    "ml", "iii", "t.j", "t. j", "spol", "mil",
    "ii", "napr", "resp", "tzv",
];

#[rustfmt::skip]
const PREPOSITIVE_ABBREVIATIONS: &[&str] = &[
    "st", "p", "dr", "mudr", "judr", "ing",
    "mgr", "bc", "drsc", "doc", "prof",
];

#[rustfmt::skip]
const NUMBER_ABBREVIATIONS: &[&str] = &[
    "č", "no", "nr",
];
const SENTENCE_STARTERS: &[&str] = &[];

#[rustfmt::skip]
const MONTHS: &[&str] = &[
    "Január", "Február", "Marec", "Apríl", "Máj", "Jún", "Júl", "August", "September", "Október", "November", "December",
    "Januára", "Februára", "Marca", "Apríla", "Mája", "Júna", "Júla", "Augusta", "Septembra", "Októbra", "Novembra", "Decembra",
];

pub(crate) const fn rules() -> Rules {
    let mut rules = Rules::standard();
    rules.abbreviations = ABBREVIATIONS;
    rules.prepositive_abbreviations = PREPOSITIVE_ABBREVIATIONS;
    rules.number_abbreviations = NUMBER_ABBREVIATIONS;
    rules.sentence_starters = SENTENCE_STARTERS;
    rules.abbreviation_mode = AbbreviationMode::Slovak;
    rules.list_mode = ListMode::NoAlphabetical;
    rules.number_mode = NumberMode::Slovak;
    rules.punctuation_mode = PunctuationMode::Slovak;
    rules.date_words = MONTHS;
    rules
}
