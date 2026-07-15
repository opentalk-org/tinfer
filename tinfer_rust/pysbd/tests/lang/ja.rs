use pysbd::{Options, Segmenter};

struct Case {
    text: &'static str,
    expected: &'static [&'static str],
}

fn assert_cases(cases: &[Case], options: Options) {
    let segmenter = Segmenter::new("ja", options).unwrap();
    for case in cases {
        let actual = segmenter.segment(case.text).unwrap();
        let actual: Vec<&str> = actual.iter().map(|sentence| sentence.trim()).collect();
        assert_eq!(actual, case.expected, "input: {}", case.text);
    }
}

const GOLDEN_JA_RULES_TEST_CASES: &[Case] = &[
    Case {
        text: r#"これはペンです。それはマーカーです。"#, expected: &[r#"これはペンです。"#, r#"それはマーカーです。"#]
    },
    Case { text: r#"それは何ですか？ペンですか？"#, expected: &[r#"それは何ですか？"#, r#"ペンですか？"#] },
    Case { text: r#"良かったね！すごい！"#, expected: &[r#"良かったね！"#, r#"すごい！"#] },
    Case {
        text: r#"自民党税制調査会の幹部は、「引き下げ幅は３．２９％以上を目指すことになる」と指摘していて、今後、公明党と合意したうえで、３０日に決定する与党税制改正大綱に盛り込むことにしています。２％台後半を目指すとする方向で最終調整に入りました。"#,
        expected: &[
            r#"自民党税制調査会の幹部は、「引き下げ幅は３．２９％以上を目指すことになる」と指摘していて、今後、公明党と合意したうえで、３０日に決定する与党税制改正大綱に盛り込むことにしています。"#,
            r#"２％台後半を目指すとする方向で最終調整に入りました。"#,
        ],
    },
];

#[test]
fn upstream_golden_ja_rules_test_cases() {
    assert_cases(GOLDEN_JA_RULES_TEST_CASES, Options { clean: false, doc_type: None });
}

const JA_TEST_CASES_CLEAN: &[Case] = &[Case {
    text: r#"これは父の
家です。"#,
    expected: &[r#"これは父の家です。"#],
}];

#[test]
fn upstream_ja_test_cases_clean() {
    assert_cases(JA_TEST_CASES_CLEAN, Options { clean: true, doc_type: None });
}
