use thiserror::Error;
use crate::phonemizer_pool::Phonemizer;

mod align;
mod espeak;
mod phonemizer_pool;
mod split;
mod tokenize;
mod utf8;

#[derive(Debug, Error)]
pub enum AlignrustError {
    #[error("{0}")]
    Message(String),
}

#[derive(Clone)]
pub struct EngineConfig {
    pub language: String,
    pub tie: bool,
}

pub struct Engine {
    cfg: EngineConfig,
    phonemizer: Option<phonemizer_pool::ForkedEspeakPool>,
    espeak_workers: usize,
}

impl Engine {
    pub fn new(language: &str, tie: bool, espeak_workers: usize) -> Self {
        Self {
            cfg: EngineConfig {
                language: language.to_owned(),
                tie,
            },
            phonemizer: None,
            espeak_workers: espeak_workers.max(1),
        }
    }

    fn ensure_phonemizer(&mut self) -> Result<(), AlignrustError> {
        if self.phonemizer.is_none() {
            self.phonemizer = Some(phonemizer_pool::ForkedEspeakPool::new(
                &self.cfg,
                self.espeak_workers,
            )?);
        }
        Ok(())
    }

    pub fn text_to_phonemes(&mut self, text: &str) -> Result<String, AlignrustError> {
        self.ensure_phonemizer()?;
        self.phonemizer
            .as_mut()
            .ok_or_else(|| AlignrustError::Message("phonemizer missing".to_owned()))?
            .text_to_phonemes(text)
    }

    pub fn align(
        &mut self,
        text: &str,
        punctuation: &str,
        threads: usize,
    ) -> Result<(Vec<String>, Vec<String>), AlignrustError> {
        self.ensure_phonemizer()?;
        let (t, p) = align::align_text_with_threads(
            self.phonemizer
                .as_ref()
                .ok_or_else(|| AlignrustError::Message("phonemizer missing".to_owned()))?,
            text,
            punctuation,
            |a, b| split::split_by_punctuation_impl(a, b),
            threads.max(1),
        );
        Ok((t, p))
    }

    pub fn align_batch(
        &mut self,
        texts: &[String],
        punctuation: &str,
        threads: usize,
    ) -> Result<Vec<(Vec<String>, Vec<String>)>, AlignrustError> {
        self.ensure_phonemizer()?;
        let phonemizer = self
            .phonemizer
            .as_ref()
            .ok_or_else(|| AlignrustError::Message("phonemizer missing".to_owned()))?;

        Ok(align::align_texts_batch_with_threads(
            phonemizer,
            texts,
            punctuation,
            |a, b| split::split_by_punctuation_impl(a, b),
            threads.max(1),
        ))
    }
}

pub fn split_by_punctuation(
    text: &str,
    punctuation: &str,
) -> Result<(Vec<String>, Vec<(i32, String)>), AlignrustError> {
    Ok(split::split_by_punctuation_impl(text, punctuation))
}
