use pysbd::{Options, Segmenter};

struct Case {
    text: &'static str,
    expected: &'static [&'static str],
}

fn assert_cases(cases: &[Case], options: Options) {
    let segmenter = Segmenter::new("hi", options).unwrap();
    for case in cases {
        let actual = segmenter.segment(case.text).unwrap();
        let actual: Vec<&str> = actual.iter().map(|sentence| sentence.trim()).collect();
        assert_eq!(actual, case.expected, "input: {}", case.text);
    }
}

const GOLDEN_HI_RULES_TEST_CASES: &[Case] = &[
    Case {
        text: r#"सच्चाई यह है कि इसे कोई नहीं जानता। हो सकता है यह फ़्रेन्को के खिलाफ़ कोई विद्रोह रहा हो, या फिर बेकाबू हो गया कोई आनंदोत्सव।"#,
        expected: &[r#"सच्चाई यह है कि इसे कोई नहीं जानता।"#, r#"हो सकता है यह फ़्रेन्को के खिलाफ़ कोई विद्रोह रहा हो, या फिर बेकाबू हो गया कोई आनंदोत्सव।"#],
    },
];

#[test]
fn upstream_golden_hi_rules_test_cases() {
    assert_cases(GOLDEN_HI_RULES_TEST_CASES, Options { clean: false, doc_type: None });
}
