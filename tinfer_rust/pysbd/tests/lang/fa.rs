use pysbd::{Options, Segmenter};

struct Case {
    text: &'static str,
    expected: &'static [&'static str],
}

#[rustfmt::skip]
const GOLDEN_FA_RULES_TEST_CASES: &[Case] = &[
    Case { text: "خوشبختم، آقای رضا. شما کجایی هستید؟ من از تهران هستم.", expected: &["خوشبختم، آقای رضا.", "شما کجایی هستید؟", "من از تهران هستم."] },
];

#[test]
fn golden_rules_match_upstream() {
    let segmenter = Segmenter::new("fa", Options { clean: false, doc_type: None }).unwrap();
    for case in GOLDEN_FA_RULES_TEST_CASES {
        let actual = segmenter.segment(case.text).unwrap();
        let trimmed: Vec<&str> = actual.iter().map(|sentence| sentence.trim()).collect();
        assert_eq!(trimmed, case.expected, "{}", case.text);
    }
}
