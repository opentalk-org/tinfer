use pysbd::{Options, Segmenter};

struct Case {
    text: &'static str,
    expected: &'static [&'static str],
}

#[rustfmt::skip]
const GOLDEN_PL_RULES_TEST_CASES: &[Case] = &[
    Case { text: "To słowo bałt. jestskrótem.", expected: &["To słowo bałt. jestskrótem."] },
];

#[test]
fn golden_rules_match_upstream() {
    let segmenter = Segmenter::new("pl", Options { clean: false, doc_type: None }).unwrap();
    for case in GOLDEN_PL_RULES_TEST_CASES {
        let actual = segmenter.segment(case.text).unwrap();
        let trimmed: Vec<&str> = actual.iter().map(|sentence| sentence.trim()).collect();
        assert_eq!(trimmed, case.expected, "{}", case.text);
    }
}
