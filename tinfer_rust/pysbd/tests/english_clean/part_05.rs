use super::Case;

pub(super) const CASES: &[Case] = &[
    Case { text: "He has Ph.D.-level training", expected: &["He has Ph.D.-level training"] },
    Case { text: "He has Ph.D. level training", expected: &["He has Ph.D. level training"] },
    Case {
        text: "I will be paid Rs. 16720/- in total for the time spent and the inconvenience caused to me, only after completion of all aspects of the study.",
        expected: &[
            "I will be paid Rs. 16720/- in total for the time spent and the inconvenience caused to me, only after completion of all aspects of the study.",
        ],
    },
    Case {
        text: "If I decide to withdraw from the study for other reasons, I will be paid only up to the extent of my participation amount according to the approved procedure of Apotex BEC. If I complete all aspects in Period 1, I will be paid Rs. 3520 and if I complete all aspects in Period 1 and Period 2, I will be paid Rs. 7790 and if I complete all aspects in Period 1, Period 2 and Period 3, I will be paid Rs. 12060 at the end of the study.",
        expected: &[
            "If I decide to withdraw from the study for other reasons, I will be paid only up to the extent of my participation amount according to the approved procedure of Apotex BEC.",
            "If I complete all aspects in Period 1, I will be paid Rs. 3520 and if I complete all aspects in Period 1 and Period 2, I will be paid Rs. 7790 and if I complete all aspects in Period 1, Period 2 and Period 3, I will be paid Rs. 12060 at the end of the study.",
        ],
    },
    Case {
        text: "After completion of each Period, I will be paid an advance amount of rs. 1000 and this amount will be deducted from my final study compensation.",
        expected: &[
            "After completion of each Period, I will be paid an advance amount of rs. 1000 and this amount will be deducted from my final study compensation.",
        ],
    },
    Case {
        text: "Mix it, put it in the oven, and -- voila! -- you have cake.",
        expected: &["Mix it, put it in the oven, and -- voila! -- you have cake."],
    },
    Case {
        text: "Some can be -- if I may say so? -- a bit questionable.",
        expected: &["Some can be -- if I may say so? -- a bit questionable."],
    },
    Case {
        text: "What do you see? - Posted like silent sentinels all around the town, stand thousands upon thousands of mortal men fixed in ocean reveries.",
        expected: &[
            "What do you see?",
            "- Posted like silent sentinels all around the town, stand thousands upon thousands of mortal men fixed in ocean reveries.",
        ],
    },
    Case {
        text: "In placebo-controlled studies of all uses of Tracleer, marked decreases in hemoglobin (>15% decrease from baseline resulting in values <11 g/ dL) were observed in 6% of Tracleer-treated patients and 3% of placebo-treated patients. Bosentan is highly bound (>98%) to plasma proteins, mainly albumin.",
        expected: &[
            "In placebo-controlled studies of all uses of Tracleer, marked decreases in hemoglobin (>15% decrease from baseline resulting in values <11 g/ dL) were observed in 6% of Tracleer-treated patients and 3% of placebo-treated patients.",
            "Bosentan is highly bound (>98%) to plasma proteins, mainly albumin.",
        ],
    },
    Case {
        text: "The parties to this Agreement are PragmaticSegmenterExampleCompanyA Inc. (“Company A”), and PragmaticSegmenterExampleCompanyB Inc. (“Company B”).",
        expected: &[
            "The parties to this Agreement are PragmaticSegmenterExampleCompanyA Inc. (“Company A”), and PragmaticSegmenterExampleCompanyB Inc. (“Company B”).",
        ],
    },
];
