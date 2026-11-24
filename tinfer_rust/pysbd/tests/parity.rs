use pysbd::{Options, Segmenter, TextSpan};

struct Case {
    language: &'static str,
    text: &'static str,
    clean: bool,
    expected: &'static [&'static str],
    spans: &'static [(&'static str, usize, usize)],
}

#[test]
fn differential_failures_match_python_034() {
    for case in CASES {
        let segmenter = Segmenter::new(case.language, Options { clean: case.clean, doc_type: None }).unwrap();
        assert_eq!(segmenter.segment(case.text).unwrap(), case.expected, "{} {:?}", case.language, case.text);
        if !case.clean {
            let expected: Vec<_> =
                case.spans.iter().map(|(text, start, end)| TextSpan { text: (*text).into(), start: *start, end: *end }).collect();
            assert_eq!(segmenter.segment_spans(case.text).unwrap(), expected, "{} {:?}", case.language, case.text);
        }
    }
}

const CASES: &[Case] = &[
    Case {
        language: "en",
        text: "From the AP comes this story :\nPresident Bush on Tuesday nominated two individuals to replace retiring jurists on federal courts in the Washington area.\n***\nAfter you are elected in 2004, what will your memoirs say about you, what will the title be, and what will the main theme say?\n***\n\"THE PRESIDENT: I appreciate that.\n(Laughter.)\nMy life is too complicated right now trying to do my job.\n(Laughter.)",
        clean: false,
        expected: &[
            "From the AP comes this story :\n",
            "President Bush on Tuesday nominated two individuals to replace retiring jurists on federal courts in the Washington area.\n",
            "***\n",
            "After you are elected in 2004, what will your memoirs say about you, what will the title be, and what will the main theme say?\n",
            "***\n",
            "\"THE PRESIDENT: I appreciate that.\n",
            "(Laughter.)\n",
            "My life is too complicated right now trying to do my job.\n",
            "(Laughter.)",
        ],
        spans: &[
            ("From the AP comes this story :\n", 0, 31),
            (
                "President Bush on Tuesday nominated two individuals to replace retiring jurists on federal courts in the Washington area.\n",
                31,
                153,
            ),
            ("***\n", 153, 157),
            (
                "After you are elected in 2004, what will your memoirs say about you, what will the title be, and what will the main theme say?\n",
                157,
                284,
            ),
            ("***\n", 284, 288),
            ("\"THE PRESIDENT: I appreciate that.\n", 288, 323),
            ("(Laughter.)\n", 323, 335),
            ("My life is too complicated right now trying to do my job.\n", 335, 393),
            ("(Laughter.)", 393, 404),
        ],
    },
    Case {
        language: "en",
        text: "From the AP comes this story :\nPresident Bush on Tuesday nominated two individuals to replace retiring jurists on federal courts in the Washington area.\n***\nAfter you are elected in 2004, what will your memoirs say about you, what will the title be, and what will the main theme say?\n***\n\"THE PRESIDENT: I appreciate that.\n(Laughter.)\nMy life is too complicated right now trying to do my job.\n(Laughter.)",
        clean: true,
        expected: &[
            "From the AP comes this story :",
            "President Bush on Tuesday nominated two individuals to replace retiring jurists on federal courts in the Washington area.",
            "***",
            "After you are elected in 2004, what will your memoirs say about you, what will the title be, and what will the main theme say?",
            "***",
            "\"THE PRESIDENT: I appreciate that.",
            "(Laughter.)",
            "My life is too complicated right now trying to do my job.",
            "(Laughter.)",
        ],
        spans: &[],
    },
    Case {
        language: "ru",
        text: "«К чему ты готовишься? – спросила мама.\n– Завтра ведь выходной».",
        clean: false,
        expected: &["«К чему ты готовишься? ", "– спросила мама.\n", "– Завтра ведь выходной»."],
        spans: &[("«К чему ты готовишься? ", 0, 23), ("– спросила мама.\n", 23, 40), ("– Завтра ведь выходной».", 40, 64)],
    },
    Case {
        language: "sk",
        text: "„Prieskumy beriem na ľahkú váhu.\nV podstate ma to nezaujíma,“ reagoval Matovič na prieskum agentúry Focus.",
        clean: false,
        expected: &["„Prieskumy beriem na ľahkú váhu.\n", "V podstate ma to nezaujíma,“ reagoval Matovič na prieskum agentúry Focus."],
        spans: &[
            ("„Prieskumy beriem na ľahkú váhu.\n", 0, 33),
            ("V podstate ma to nezaujíma,“ reagoval Matovič na prieskum agentúry Focus.", 33, 106),
        ],
    },
    Case {
        language: "en",
        text: "Maissen se chargea du reste .... Logiquement,",
        clean: false,
        expected: &["Maissen se chargea du reste .", "...", "Logiquement,"],
        spans: &[("Maissen se chargea du reste .", 0, 29), ("...", 28, 31), ("Logiquement,", 33, 45)],
    },
    Case { language: "en", text: "foo ∯ bar. Next.", clean: false, expected: &["Next."], spans: &[("Next.", 11, 16)] },
    Case { language: "en", text: "foo ∮ bar. Next.", clean: false, expected: &["Next."], spans: &[("Next.", 11, 16)] },
    Case {
        language: "en",
        text: "foo ☉ bar. Next.",
        clean: false,
        expected: &["bar. ", "Next."],
        spans: &[("bar. ", 6, 11), ("Next.", 11, 16)],
    },
    Case { language: "en", text: "foo ♨ bar. Next.", clean: false, expected: &["Next."], spans: &[("Next.", 11, 16)] },
    Case { language: "en", text: "foo ♨ bar. Next.", clean: true, expected: &["foo . bar.", "Next."], spans: &[] },
    Case { language: "en", text: "foo &⎋& bar. Next.", clean: false, expected: &["Next."], spans: &[("Next.", 13, 18)] },
    Case {
        language: "en",
        text: "Hello.\u{001c}Goodbye.",
        clean: false,
        expected: &["Hello.\u{001c}", "Goodbye."],
        spans: &[("Hello.\u{001c}", 0, 7), ("Goodbye.", 7, 15)],
    },
    Case { language: "en", text: "Hello.\u{001c}Goodbye.", clean: true, expected: &["Hello.", "Goodbye."], spans: &[] },
    Case {
        language: "en",
        text: "Hello.\u{001d}Goodbye.",
        clean: false,
        expected: &["Hello.\u{001d}", "Goodbye."],
        spans: &[("Hello.\u{001d}", 0, 7), ("Goodbye.", 7, 15)],
    },
    Case { language: "en", text: "Hello.\u{001d}Goodbye.", clean: true, expected: &["Hello.", "Goodbye."], spans: &[] },
    Case {
        language: "en",
        text: "Hello.\u{001e}Goodbye.",
        clean: false,
        expected: &["Hello.\u{001e}", "Goodbye."],
        spans: &[("Hello.\u{001e}", 0, 7), ("Goodbye.", 7, 15)],
    },
    Case { language: "en", text: "Hello.\u{001e}Goodbye.", clean: true, expected: &["Hello.", "Goodbye."], spans: &[] },
    Case {
        language: "en",
        text: "Hello.\u{001f}Goodbye.",
        clean: false,
        expected: &["Hello.\u{001f}", "Goodbye."],
        spans: &[("Hello.\u{001f}", 0, 7), ("Goodbye.", 7, 15)],
    },
    Case { language: "en", text: "Hello.\u{001f}Goodbye.", clean: true, expected: &["Hello.", "Goodbye."], spans: &[] },
    Case {
        language: "en",
        text: "\"aaaaaaaaaaaaaaaaaaaaaaaa. Next.",
        clean: false,
        expected: &["\"aaaaaaaaaaaaaaaaaaaaaaaa. ", "Next."],
        spans: &[("\"aaaaaaaaaaaaaaaaaaaaaaaa. ", 0, 27), ("Next.", 27, 32)],
    },
];
