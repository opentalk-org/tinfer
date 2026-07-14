use pysbd::{DocType, Options, Rule, Segmenter, apply_rules, python_replacement};

fn plain_options() -> Options {
    Options { clean: false, doc_type: None }
}

#[test]
fn empty_input_has_no_segments_or_spans() {
    let segmenter = Segmenter::new("en", plain_options()).unwrap();

    assert!(segmenter.segment("").unwrap().is_empty());
    assert!(segmenter.segment("\n").unwrap().is_empty());
    assert!(segmenter.segment_spans("").unwrap().is_empty());
    assert!(segmenter.segment_spans("\n").unwrap().is_empty());
}

#[test]
fn invalid_option_combinations_are_rejected() {
    let pdf_without_cleaning = Options { clean: false, doc_type: Some(DocType::Pdf) };
    assert_eq!(
        Segmenter::new("en", pdf_without_cleaning).unwrap_err().to_string(),
        "`doc_type='pdf'` should have `clean=True` & `char_span` should be False since originaltext will be modified."
    );

    let clean_segmenter = Segmenter::new("en", Options { clean: true, doc_type: None }).unwrap();
    assert_eq!(
        clean_segmenter.segment_spans("A sentence.").unwrap_err().to_string(),
        "char_span must be False if clean is True. Since `clean=True` will modify original text."
    );
}

#[test]
fn unsupported_language_precedes_option_validation() {
    let result = Segmenter::new("xx", Options { clean: false, doc_type: Some(DocType::Pdf) });

    assert_eq!(result.unwrap_err().to_string(), "unsupported language code: xx");
}

#[test]
fn spans_preserve_whitespace_and_character_offsets() {
    let text = "My name is Jonas E. Smith. Please turn to p. 55.  \n";
    let segmenter = Segmenter::new("en", plain_options()).unwrap();
    let spans = segmenter.segment_spans(text).unwrap();

    assert_eq!(spans.iter().map(|span| span.text.as_str()).collect::<String>(), text);
    assert_eq!(spans.iter().map(|span| span.text.as_str()).collect::<String>(), text);

    let unicode = "नमस्ते। 你好。";
    let span = segmenter.segment_spans(unicode).unwrap().remove(0);
    assert_eq!((span.start, span.end), (0, unicode.chars().count()));
    assert_eq!(span.text, unicode);
}

#[test]
fn segmentation_is_non_destructive() {
    let text = "My name is Jonas E. Smith. Please turn to p. 55.";
    let segmenter = Segmenter::new("en", plain_options()).unwrap();

    assert_eq!(segmenter.segment(text).unwrap().concat(), text);
    assert_eq!(text, "My name is Jonas E. Smith. Please turn to p. 55.");
}

#[test]
fn rules_apply_in_order_with_python_groups() {
    let rules = [Rule::new(r"(Jonas) (Smith)", r"\2, \1"), Rule::new(r"(?<=Smith),", "!")];

    assert_eq!(apply_rules("Jonas Smith", &rules).unwrap(), "Smith! Jonas");
    assert_eq!(python_replacement(r"\g<name> \1 $$"), "${name} $1 $$$$");
}

#[test]
fn invalid_regex_is_an_explicit_error() {
    let invalid = Rule::new("(", "replacement");

    assert!(invalid.replace_all("text").is_err());
}
