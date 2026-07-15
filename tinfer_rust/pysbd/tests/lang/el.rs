use pysbd::{Options, Segmenter};

struct Case {
    text: &'static str,
    expected: &'static [&'static str],
}

fn assert_cases(cases: &[Case], options: Options) {
    let segmenter = Segmenter::new("el", options).unwrap();
    for case in cases {
        let actual = segmenter.segment(case.text).unwrap();
        let actual: Vec<&str> = actual.iter().map(|sentence| sentence.trim()).collect();
        assert_eq!(actual, case.expected, "input: {}", case.text);
    }
}

const GOLDEN_EL_RULES_TEST_CASES: &[Case] = &[Case {
    text: r#"Με συγχωρείτε· πού είναι οι τουαλέτες; Τις Κυριακές δε δούλευε κανένας. το κόστος του σπιτιού ήταν £260.950,00."#,
    expected: &[
        r#"Με συγχωρείτε· πού είναι οι τουαλέτες;"#,
        r#"Τις Κυριακές δε δούλευε κανένας."#,
        r#"το κόστος του σπιτιού ήταν £260.950,00."#,
    ],
}];

#[test]
fn upstream_golden_el_rules_test_cases() {
    assert_cases(GOLDEN_EL_RULES_TEST_CASES, Options { clean: false, doc_type: None });
}
