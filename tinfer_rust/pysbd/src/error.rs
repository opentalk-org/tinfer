use thiserror::Error;

#[derive(Debug, Error)]
pub enum Error {
    #[error("unsupported language code: {0}")]
    UnsupportedLanguage(String),
    #[error("{0}")]
    InvalidOptions(&'static str),
    #[error("regex `{pattern}` failed: {message}")]
    Regex { pattern: String, message: String },
    #[error("sentence `{sentence}` is not contiguous at byte offset {offset}")]
    Span { sentence: String, offset: usize },
}

pub type Result<T> = std::result::Result<T, Error>;
