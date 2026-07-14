use fancy_regex::{Captures, Regex};

use crate::{Error, Result, Rules, punctuation};

pub(crate) const fn rules() -> Rules {
    Rules::standard()
}

pub(crate) fn between(text: &str) -> Result<String> {
    let mut output = text.to_owned();
    for pattern in [r"《(?=(?P<tmp>[^》\\]+|\\{2}|\\.)*)(?P=tmp)》", r"「(?=(?P<tmp>[^」\\]+|\\{2}|\\.)*)(?P=tmp)」"] {
        let regex = Regex::new(pattern).map_err(|error| Error::Regex { pattern: pattern.into(), message: error.to_string() })?;
        output = regex
            .try_replacen(&output, 0, |captures: &Captures<'_>| punctuation::protect(&captures[0], true))
            .map_err(|error| Error::Regex { pattern: pattern.into(), message: error.to_string() })?
            .into_owned();
    }
    Ok(output)
}
