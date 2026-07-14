use pysbd::{Options, Segmenter};

struct Case {
    text: &'static str,
    expected: &'static [&'static str],
}

#[rustfmt::skip]
const GOLDEN_SK_RULES_TEST_CASES: &[Case] = &[
    Case { text: "Ide o majiteľov firmy ABTrade s. r. o., ktorí stoja aj za ďalšími spoločnosťami, napr. XYZCorp a.s.", expected: &["Ide o majiteľov firmy ABTrade s. r. o., ktorí stoja aj za ďalšími spoločnosťami, napr. XYZCorp a.s."] },
    Case { text: "„Prieskumy beriem na ľahkú váhu. V podstate ma to nezaujíma,“ reagoval Matovič na prieskum agentúry Focus.", expected: &["„Prieskumy beriem na ľahkú váhu. V podstate ma to nezaujíma,“ reagoval Matovič na prieskum agentúry Focus."] },
    Case { text: "Toto sa mi podarilo až na 10. pokus, ale stálo to za to.", expected: &["Toto sa mi podarilo až na 10. pokus, ale stálo to za to."] },
    Case { text: "Ide o príslušníkov XII. Pluku špeciálneho určenia.", expected: &["Ide o príslušníkov XII. Pluku špeciálneho určenia."] },
    Case { text: "Spoločnosť bola založená 7. Apríla 2020, na zmluve však figuruje dátum 20. marec 2020.", expected: &["Spoločnosť bola založená 7. Apríla 2020, na zmluve však figuruje dátum 20. marec 2020."] },
];

#[test]
fn golden_rules_match_upstream() {
    let segmenter = Segmenter::new("sk", Options { clean: false, doc_type: None }).unwrap();
    for case in GOLDEN_SK_RULES_TEST_CASES {
        let actual = segmenter.segment(case.text).unwrap();
        let trimmed: Vec<&str> = actual.iter().map(|sentence| sentence.trim()).collect();
        assert_eq!(trimmed, case.expected, "{}", case.text);
    }
}
