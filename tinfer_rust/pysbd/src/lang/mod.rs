mod am;
mod ar;
mod bg;
mod da;
mod de;
mod el;
mod en;
mod es;
mod fa;
mod fr;
mod hi;
mod hy;
mod it;
mod ja;
mod kk;
mod mr;
mod my;
mod nl;
mod pl;
mod ru;
mod sk;
mod ur;
mod zh;

pub(crate) use en::rules as english_rules;

use crate::{BoundaryMode, Language, PunctuationMode, Result, Rule, Rules, punctuation};

pub(crate) const fn rules(language: Language) -> Rules {
    match language {
        Language::Amharic => am::rules(),
        Language::Arabic => ar::rules(),
        Language::Armenian => hy::rules(),
        Language::Bulgarian => bg::rules(),
        Language::Burmese => my::rules(),
        Language::Chinese => zh::rules(),
        Language::Danish => da::rules(),
        Language::Dutch => nl::rules(),
        Language::English => en::rules(),
        Language::French => fr::rules(),
        Language::German => de::rules(),
        Language::Greek => el::rules(),
        Language::Hindi => hi::rules(),
        Language::Italian => it::rules(),
        Language::Japanese => ja::rules(),
        Language::Kazakh => kk::rules(),
        Language::Marathi => mr::rules(),
        Language::Persian => fa::rules(),
        Language::Polish => pl::rules(),
        Language::Russian => ru::rules(),
        Language::Slovak => sk::rules(),
        Language::Spanish => es::rules(),
        Language::Urdu => ur::rules(),
    }
}

pub(crate) fn clean(text: &str, language: Language) -> Result<String> {
    match language {
        Language::Japanese => ja::clean(text),
        _ => Ok(text.to_owned()),
    }
}

pub(crate) fn replace_numbers(text: &str, language: Language) -> Result<String> {
    match language {
        Language::Danish => da::replace_numbers(text),
        Language::German => de::replace_numbers(text),
        _ => Ok(text.to_owned()),
    }
}

pub(crate) fn between_punctuation(text: &str, language: Language, mode: PunctuationMode) -> Result<String> {
    match language {
        Language::Chinese => zh::between(text),
        Language::German => de::between_punctuation(text),
        Language::Japanese => ja::between_punctuation(text),
        Language::Kazakh => kk::between_punctuation(text),
        _ => punctuation::between(text, mode),
    }
}

pub(crate) fn before_boundaries(text: &str, language: Language, mode: BoundaryMode) -> Result<String> {
    let mut output = match language {
        Language::Arabic => ar::before_boundaries(text)?,
        _ => Rule::new(r"&ᓴ&$", "!").replace_all(text)?,
    };
    if mode == BoundaryMode::Persian {
        output = Rule::new(r"(?<=\d):(?=\d)", "♭").replace_all(&output)?;
        output = Rule::new(r"،(?=\s\S+،)", "♬").replace_all(&output)?;
    }
    Ok(output)
}

pub(crate) fn restore_abbreviation_boundaries(text: &str, language: Language, starters: &[&str]) -> Result<String> {
    if language == Language::Danish {
        return da::restore_abbreviation_boundaries(text);
    }
    if starters.is_empty() {
        return Ok(text.to_owned());
    }
    let starters = starters.iter().map(|word| regex::escape(word)).collect::<Vec<_>>().join("|");
    let pattern = format!(r"(U∯S|U\.S|U∯K|E∯U|E\.U|U∯S∯A|U\.S\.A|I|i.v|I.V)∯(?=\s(?:{starters})\s)");
    Rule::new(pattern, r"\1.").replace_all(text)
}
