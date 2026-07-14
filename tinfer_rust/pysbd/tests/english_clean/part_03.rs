use super::Case;

pub(super) const CASES: &[Case] = &[
    Case { text: "SEC. 1262 AUTHORIZATION OF APPROPRIATIONS.", expected: &["SEC. 1262 AUTHORIZATION OF APPROPRIATIONS."] },
    Case { text: "a", expected: &["a"] },
    Case {
        text: "I wrote this in the 'nineties.  It has four sentences.  This is the third, isn't it?  And this is the last",
        expected: &["I wrote this in the 'nineties.", "It has four sentences.", "This is the third, isn't it?", "And this is the last"],
    },
    Case {
        text: "I wrote this in the ’nineties.  It has four sentences.  This is the third, isn't it?  And this is the last",
        expected: &["I wrote this in the ’nineties.", "It has four sentences.", "This is the third, isn't it?", "And this is the last"],
    },
    Case {
        text: "Unlike the abbreviations i.e. and e.g., viz. is used to indicate a detailed description of something stated before.",
        expected: &["Unlike the abbreviations i.e. and e.g., viz. is used to indicate a detailed description of something stated before."],
    },
    Case {
        text: "For example, ‘dragonswort… is said that it should be grown in dragon’s blood. It grows at the tops of mountains where there are groves of trees, chiefly in holy places and in the country that is called Apulia’ (translated by Anne Van Arsdall, in Medieval Herbal Remedies: The Old English Herbarium and Anglo-Saxon Medicine p. 154). The Herbal also includes lore about other plants, such as the mandrake.",
        expected: &[
            "For example, ‘dragonswort… is said that it should be grown in dragon’s blood. It grows at the tops of mountains where there are groves of trees, chiefly in holy places and in the country that is called Apulia’ (translated by Anne Van Arsdall, in Medieval Herbal Remedies: The Old English Herbarium and Anglo-Saxon Medicine p. 154).",
            "The Herbal also includes lore about other plants, such as the mandrake.",
        ],
    },
    Case {
        text: "Here’s the - ahem - official citation: Baker, C., Anderson, Kenneth, Martin, James, & Palen, Leysia. Modeling Open Source Software Communities, ProQuest Dissertations and Theses.",
        expected: &[
            "Here’s the - ahem - official citation: Baker, C., Anderson, Kenneth, Martin, James, & Palen, Leysia.",
            "Modeling Open Source Software Communities, ProQuest Dissertations and Theses.",
        ],
    },
    Case {
        text: "These include images of various modes of transport and members of the team, all available in .jpeg format. Images can be downloaded from our website. We also offer archives as .zip files.",
        expected: &[
            "These include images of various modes of transport and members of the team, all available in .jpeg format.",
            "Images can be downloaded from our website.",
            "We also offer archives as .zip files.",
        ],
    },
    Case {
        text: "Saint Maximus (died 250) is a Christian saint and martyr.[1] The emperor Decius published a decree ordering the veneration of busts of the deified emperors.",
        expected: &[
            "Saint Maximus (died 250) is a Christian saint and martyr.[1]",
            "The emperor Decius published a decree ordering the veneration of busts of the deified emperors.",
        ],
    },
    Case {
        text: "Differing agendas can potentially create an understanding gap in a consultation.11 12 Take the example of one of the most common presentations in ill health: the common cold.",
        expected: &[
            "Differing agendas can potentially create an understanding gap in a consultation.11 12",
            "Take the example of one of the most common presentations in ill health: the common cold.",
        ],
    },
    Case {
        text: "Daniel Kahneman popularised the concept of fast and slow thinking: the distinction between instinctive (type 1 thinking) and reflective, analytical cognition (type 2).10 This model relates to doctors achieving a balance between efficiency and effectiveness.",
        expected: &[
            "Daniel Kahneman popularised the concept of fast and slow thinking: the distinction between instinctive (type 1 thinking) and reflective, analytical cognition (type 2).10",
            "This model relates to doctors achieving a balance between efficiency and effectiveness.",
        ],
    },
    Case {
        text: "Its traditional use[1] is well documented in the ethnobotanical literature [2–11]. Leaves, buds, tar and essential oils are used to treat a wide spectrum of diseases.",
        expected: &[
            "Its traditional use[1] is well documented in the ethnobotanical literature [2–11].",
            "Leaves, buds, tar and essential oils are used to treat a wide spectrum of diseases.",
        ],
    },
    Case {
        text: "Thus increasing the desire for political reform both in Lancashire and in the country at large.[7][8] This was a serious misdemeanour,[16] encouraging them to declare the assembly illegal as soon as it was announced on 31 July.[17][18] The radicals sought a second opinion on the meeting's legality.",
        expected: &[
            "Thus increasing the desire for political reform both in Lancashire and in the country at large.[7][8]",
            "This was a serious misdemeanour,[16] encouraging them to declare the assembly illegal as soon as it was announced on 31 July.[17][18]",
            "The radicals sought a second opinion on the meeting's legality.",
        ],
    },
    Case {
        text: "The table in (4) is a sample from the Wall Street Journal (1987).1 According to the distribution all the pairs given in (4) count as candidates for abbreviations.",
        expected: &[
            "The table in (4) is a sample from the Wall Street Journal (1987).1",
            "According to the distribution all the pairs given in (4) count as candidates for abbreviations.",
        ],
    },
];
