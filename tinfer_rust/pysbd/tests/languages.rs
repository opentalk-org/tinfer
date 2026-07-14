use pysbd::{Language, Options, Segmenter};

#[test]
fn unsupported_language_is_rejected() {
    let result = Segmenter::new("xx", Options { clean: false, doc_type: None });

    assert!(result.is_err());
}

#[test]
fn every_upstream_language_code_is_registered() {
    let codes = [
        "en", "hi", "mr", "zh", "es", "am", "ar", "hy", "bg", "ur", "ru", "pl", "fa", "nl", "da", "fr", "my", "el", "it", "ja", "de", "kk",
        "sk",
    ];

    for code in codes {
        assert!(Language::from_code(code).is_ok(), "missing language {code}");
    }
}
