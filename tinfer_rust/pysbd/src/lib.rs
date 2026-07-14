mod abbreviation;
mod cleaner;
mod error;
mod lang;
mod lists;
mod processor;
mod punctuation;
mod rules;
mod text;

pub use cleaner::clean;
pub use error::{Error, Result};
use rules::{AbbreviationMode, BoundaryMode, ListMode, NumberMode, PunctuationMode};
pub use rules::{Rule, Rules, apply_rules, python_replacement};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Options {
    pub clean: bool,
    pub doc_type: Option<DocType>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DocType {
    Pdf,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TextSpan {
    pub text: String,
    pub start: usize,
    pub end: usize,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Language {
    English,
    Hindi,
    Marathi,
    Chinese,
    Spanish,
    Amharic,
    Arabic,
    Armenian,
    Bulgarian,
    Urdu,
    Russian,
    Polish,
    Persian,
    Dutch,
    Danish,
    French,
    Burmese,
    Greek,
    Italian,
    Japanese,
    German,
    Kazakh,
    Slovak,
}

impl Language {
    pub fn from_code(code: &str) -> Result<Self> {
        match code {
            "en" => Ok(Self::English),
            "hi" => Ok(Self::Hindi),
            "mr" => Ok(Self::Marathi),
            "zh" => Ok(Self::Chinese),
            "es" => Ok(Self::Spanish),
            "am" => Ok(Self::Amharic),
            "ar" => Ok(Self::Arabic),
            "hy" => Ok(Self::Armenian),
            "bg" => Ok(Self::Bulgarian),
            "ur" => Ok(Self::Urdu),
            "ru" => Ok(Self::Russian),
            "pl" => Ok(Self::Polish),
            "fa" => Ok(Self::Persian),
            "nl" => Ok(Self::Dutch),
            "da" => Ok(Self::Danish),
            "fr" => Ok(Self::French),
            "my" => Ok(Self::Burmese),
            "el" => Ok(Self::Greek),
            "it" => Ok(Self::Italian),
            "ja" => Ok(Self::Japanese),
            "de" => Ok(Self::German),
            "kk" => Ok(Self::Kazakh),
            "sk" => Ok(Self::Slovak),
            _ => Err(Error::UnsupportedLanguage(code.to_owned())),
        }
    }

    pub const fn rules(self) -> Rules {
        lang::rules(self)
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Segmenter {
    language: Language,
    options: Options,
}

impl Segmenter {
    pub fn new(language: &str, options: Options) -> Result<Self> {
        let language = Language::from_code(language)?;
        if options.doc_type == Some(DocType::Pdf) && !options.clean {
            return Err(Error::InvalidOptions(
                "`doc_type='pdf'` should have `clean=True` & `char_span` should be False since originaltext will be modified.",
            ));
        }
        Ok(Self { language, options })
    }

    pub fn segment(&self, text: &str) -> Result<Vec<String>> {
        if text.trim().is_empty() {
            return Ok(Vec::new());
        }
        let prepared = if self.options.clean { cleaner::clean(text, self.language, self.options.doc_type)? } else { text.to_owned() };
        let sentences = processor::process(&prepared, self.language)?;
        if self.options.clean {
            return Ok(sentences);
        }
        Ok(text::locate_spans(text, &sentences)?.into_iter().map(|span| span.text).collect())
    }

    pub fn segment_spans(&self, text: &str) -> Result<Vec<TextSpan>> {
        if self.options.clean {
            return Err(Error::InvalidOptions("char_span must be False if clean is True. Since `clean=True` will modify original text."));
        }
        let sentences = self.segment(text)?;
        if sentences.is_empty() {
            return Ok(Vec::new());
        }
        text::locate_spans(text, &sentences)
    }
}
