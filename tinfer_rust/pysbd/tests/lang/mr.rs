use pysbd::{Options, Segmenter};

struct Case {
    text: &'static str,
    expected: &'static [&'static str],
}

#[rustfmt::skip]
const GOLDEN_MR_RULES_TEST_CASES: &[Case] = &[
    Case { text: "आज दसरा आहे. आज खूप शुभ दिवस आहे.", expected: &["आज दसरा आहे.", "आज खूप शुभ दिवस आहे."] },
    Case { text: "ढग खूप गर्जत होते; पण पाऊस पडत नव्हता.", expected: &["ढग खूप गर्जत होते; पण पाऊस पडत नव्हता."] },
    Case { text: "रमाची परीक्षा कधी आहे? अवकाश आहे अजून.", expected: &["रमाची परीक्षा कधी आहे?", "अवकाश आहे अजून."] },
    Case { text: "शाब्बास, असाच अभ्यास कर! आणि मग तुला नक्की यश मिळणार.", expected: &["शाब्बास, असाच अभ्यास कर!", "आणि मग तुला नक्की यश मिळणार."] },
    Case { text: "\"आपली आपण करी स्तुती तो एक मूर्ख\" असे समर्थ रामदासस्वामी म्हणतात.", expected: &["\"आपली आपण करी स्तुती तो एक मूर्ख\" असे समर्थ रामदासस्वामी म्हणतात."] },
];

#[test]
fn golden_rules_match_upstream() {
    let segmenter = Segmenter::new("mr", Options { clean: false, doc_type: None }).unwrap();
    for case in GOLDEN_MR_RULES_TEST_CASES {
        let actual = segmenter.segment(case.text).unwrap();
        let trimmed: Vec<&str> = actual.iter().map(|sentence| sentence.trim()).collect();
        assert_eq!(trimmed, case.expected, "{}", case.text);
    }
}
