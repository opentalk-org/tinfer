use pyo3::prelude::*;
use espeak_align_core as espeak_align_core;

#[pyclass]
struct Engine {
    inner: espeak_align_core::Engine,
}

#[pymethods]
impl Engine {
    #[new]
    #[pyo3(signature = (language="pl".to_owned(), tie=true, espeak_workers=4))]
    fn new(language: String, tie: bool, espeak_workers: usize) -> Self {
        Self {
            inner: espeak_align_core::Engine::new(&language, tie, espeak_workers),
        }
    }

    fn text_to_phonemes(&mut self, _py: Python<'_>, text: &str) -> PyResult<String> {
        self.inner
            .text_to_phonemes(text)
            .map_err(|e| pyo3::exceptions::PyNotImplementedError::new_err(e.to_string()))
    }

    #[pyo3(signature = (text, punctuation = r#";:,.!?¡¿—…\"«»\"\""#.to_owned(), threads=8))]
    fn align(
        &mut self,
        _py: Python<'_>,
        text: &str,
        punctuation: String,
        threads: usize,
    ) -> PyResult<(Vec<String>, Vec<String>)> {
        self.inner
            .align(text, &punctuation, threads)
            .map_err(|e| pyo3::exceptions::PyNotImplementedError::new_err(e.to_string()))
    }

    #[pyo3(signature = (texts, punctuation = r#";:,.!?¡¿—…\"«»\"\""#.to_owned(), threads=8))]
    fn align_batch(
        &mut self,
        _py: Python<'_>,
        texts: Vec<String>,
        punctuation: String,
        threads: usize,
    ) -> PyResult<Vec<(Vec<String>, Vec<String>)>> {
        self.inner
            .align_batch(&texts, &punctuation, threads)
            .map_err(|e| pyo3::exceptions::PyNotImplementedError::new_err(e.to_string()))
    }
}

#[pyfunction]
fn split_by_punctuation(text: &str, punctuation: &str) -> PyResult<(Vec<String>, Vec<(i32, String)>)> {
    espeak_align_core::split_by_punctuation(text, punctuation)
        .map_err(|e| pyo3::exceptions::PyNotImplementedError::new_err(e.to_string()))
}

#[pymodule]
fn espeak_align(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Engine>()?;
    m.add_function(wrap_pyfunction!(split_by_punctuation, m)?)?;
    Ok(())
}
