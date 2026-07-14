use crate::{Result, Rule, Rules};

const ABBREVIATIONS: &[&str] = &["Ä", "ä", "adj", "adm", "adv", "art", "asst", "b.a", "b.s", "bart", "bldg", "brig", "bros", "bse", "buchst", "bzgl", "bzw", "c.-à-d", "ca", "capt", "chr", "cmdr", "co", "col", "comdr", "con", "corp", "cpl", "d.h", "d.j", "dergl", "dgl", "dkr", "dr ", "ens", "etc", "ev ", "evtl", "ff", "g.g.a", "g.u", "gen", "ggf", "gov", "hon", "hosp", "i.f", "i.h.v", "ii", "iii", "insp", "iv", "ix", "jun", "k.o", "kath ", "lfd", "lt", "ltd", "m.e", "maj", "med", "messrs", "mio", "mlle", "mm", "mme", "mr", "mrd", "mrs", "ms", "msgr", "mwst", "no", "nos", "nr", "o.ä", "op", "ord", "pfc", "ph", "pp", "prof", "pvt", "rep", "reps", "res", "rev", "rt", "s.p.a", "sa", "sen", "sens", "sfc", "sgt", "sog", "sogen", "spp", "sr", "st", "std", "str  ", "supt", "surg", "u.a  ", "u.e", "u.s.w", "u.u", "u.ä", "usf", "usw", "v", "vgl", "vi", "vii", "viii", "vs", "x", "xi", "xii", "xiii", "xiv", "xix", "xv", "xvi", "xvii", "xviii", "xx", "z.b", "z.t", "z.z", "z.zt", "zt", "zzt", "univ.-prof", "o.univ.-prof", "ao.univ.prof", "ass.prof", "hon.prof", "univ.-doz", "univ.ass", "stud.ass", "projektass", "ass", "di", "dipl.-ing", "mag"];
const PREPOSITIVE_ABBREVIATIONS: &[&str] = &[];
const NUMBER_ABBREVIATIONS: &[&str] = &["art", "ca", "no", "nos", "nr", "pp"];
const SENTENCE_STARTERS: &[&str] = &["Am", "Auch", "Auf", "Bei", "Da", "Das", "Der", "Die", "Ein", "Eine", "Es", "Für", "Heute", "Ich", "Im", "In", "Ist", "Jetzt", "Mein", "Mit", "Nach", "So", "Und", "Warum", "Was", "Wenn", "Wer", "Wie", "Wir"];
const ABBREVIATION_RULES: &[(&str, &str)] = &[(r"(?<=\s[a-z])\.(?=\s)", "∯"), (r"(?<=^[a-z])\.(?=\s)", "∯")];
const MONTHS: &[&str] = &["Januar", "Februar", "März", "April", "Mai", "Juni", "Juli", "August", "September", "Oktober", "November", "Dezember"];

pub(crate) const fn rules() -> Rules {
    let mut rules = Rules::standard();
    rules.abbreviations = ABBREVIATIONS;
    rules.prepositive_abbreviations = PREPOSITIVE_ABBREVIATIONS;
    rules.number_abbreviations = NUMBER_ABBREVIATIONS;
    rules.sentence_starters = SENTENCE_STARTERS;
    rules.abbreviation_rules = ABBREVIATION_RULES;
    rules.abbreviations_before_uppercase = true;
    rules
}

pub(crate) fn replace_numbers(text: &str) -> Result<String> {
    let mut output = text.to_owned();
    for pattern in [r"(?<=\s\d)\.(?=\s)", r"(?<=\s\d\d)\.(?=\s)", r"(?<=-\d)\.(?=\s)", r"(?<=-\d\d)\.(?=\s)"] {
        output = Rule::new(pattern, "∯").replace_all(&output)?;
    }
    for month in MONTHS {
        output = Rule::new(format!(r"(?<=\d)\.(?=\s*{})", regex::escape(month)), "∯").replace_all(&output)?;
    }
    Ok(output)
}

pub(crate) fn between_punctuation(text: &str) -> Result<String> {
    let output = crate::punctuation::between_without_slanted_quotes(text)?;
    if output.contains('„') {
        crate::punctuation::protect_pattern(&output, r"„[^“]*“", true)
    } else if output.contains(",,") {
        crate::punctuation::protect_pattern(&output, r",,[^“]*“", true)
    } else {
        Ok(output)
    }
}
