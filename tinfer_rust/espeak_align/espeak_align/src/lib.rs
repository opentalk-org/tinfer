use espeak_align_core;
use pyo3::prelude::*;
use pyo3::types::PyDict;

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

    #[pyo3(signature = (text, punctuation = r#";:,.!?¡¿—…\"«»\"\""#.to_owned(), threads=8))]
    fn align_with_spans(
        &mut self,
        py: Python<'_>,
        text: &str,
        punctuation: String,
        threads: usize,
    ) -> PyResult<Vec<PyObject>> {
        let spans = self
            .inner
            .align_with_spans(text, &punctuation, threads)
            .map_err(|e| pyo3::exceptions::PyNotImplementedError::new_err(e.to_string()))?;

        let mut out = Vec::with_capacity(spans.len());
        for span in spans {
            let dict = PyDict::new(py);
            dict.set_item("token", span.token)?;
            dict.set_item("phonemes", span.phonemes)?;
            dict.set_item("start", span.start)?;
            dict.set_item("end", span.end)?;
            out.push(dict.into());
        }
        Ok(out)
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
fn split_by_punctuation(
    text: &str,
    punctuation: &str,
) -> PyResult<(Vec<String>, Vec<(i32, String)>)> {
    espeak_align_core::split_by_punctuation(text, punctuation)
        .map_err(|e| pyo3::exceptions::PyNotImplementedError::new_err(e.to_string()))
}

#[pymodule]
fn espeak_align(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Engine>()?;
    m.add_function(wrap_pyfunction!(split_by_punctuation, m)?)?;
    Ok(())
}
