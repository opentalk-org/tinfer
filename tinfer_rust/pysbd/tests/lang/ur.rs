use pysbd::{Options, Segmenter};

struct Case {
    text: &'static str,
    expected: &'static [&'static str],
}

#[rustfmt::skip]
const GOLDEN_UR_RULES_TEST_CASES: &[Case] = &[
    Case { text: "کیا حال ہے؟ ميرا نام ___ ەے۔ میں حالا تاوان دےدوں؟", expected: &["کیا حال ہے؟", "ميرا نام ___ ەے۔", "میں حالا تاوان دےدوں؟"] },
];

#[test]
fn golden_rules_match_upstream() {
    let segmenter = Segmenter::new("ur", Options { clean: false, doc_type: None }).unwrap();
    for case in GOLDEN_UR_RULES_TEST_CASES {
        let actual = segmenter.segment(case.text).unwrap();
        let trimmed: Vec<&str> = actual.iter().map(|sentence| sentence.trim()).collect();
        assert_eq!(trimmed, case.expected, "{}", case.text);
    }
}
