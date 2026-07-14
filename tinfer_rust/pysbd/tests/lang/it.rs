use pysbd::{Options, Segmenter};

struct Case {
    text: &'static str,
    expected: &'static [&'static str],
}

fn assert_cases(cases: &[Case], options: Options) {
    let segmenter = Segmenter::new("it", options).unwrap();
    for case in cases {
        let actual = segmenter.segment(case.text).unwrap();
        let actual: Vec<&str> = actual.iter().map(|sentence| sentence.trim()).collect();
        assert_eq!(actual, case.expected, "input: {}", case.text);
    }
}

const GOLDEN_IT_RULES_TEST_CASES: &[Case] = &[
    Case {
        text: r#"Salve Sig.ra Mengoni! Come sta oggi?"#,
        expected: &[r#"Salve Sig.ra Mengoni!"#, r#"Come sta oggi?"#],
    },
    Case {
        text: r#"Una lettera si può iniziare in questo modo «Il/la sottoscritto/a.»."#,
        expected: &[r#"Una lettera si può iniziare in questo modo «Il/la sottoscritto/a.»."#],
    },
    Case {
        text: r#"La casa costa 170.500.000,00€!"#,
        expected: &[r#"La casa costa 170.500.000,00€!"#],
    },
];

#[test]
fn upstream_golden_it_rules_test_cases() {
    assert_cases(GOLDEN_IT_RULES_TEST_CASES, Options { clean: false, doc_type: None });
}

const IT_MORE_TEST_CASES: &[Case] = &[
    Case {
        text: r#"Salve Sig.ra Mengoni! Come sta oggi?"#,
        expected: &[r#"Salve Sig.ra Mengoni!"#, r#"Come sta oggi?"#],
    },
    Case {
        text: r#"Buongiorno! Sono l'Ing. Mengozzi. È presente l'Avv. Cassioni?"#,
        expected: &[r#"Buongiorno!"#, r#"Sono l'Ing. Mengozzi."#, r#"È presente l'Avv. Cassioni?"#],
    },
    Case {
        text: r#"Mi fissi un appuntamento per mar. 23 Nov.. Grazie."#,
        expected: &[r#"Mi fissi un appuntamento per mar. 23 Nov.."#, r#"Grazie."#],
    },
    Case {
        text: r#"Ecco il mio tel.:01234567. Mi saluti la Sig.na Manelli. Arrivederci."#,
        expected: &[r#"Ecco il mio tel.:01234567."#, r#"Mi saluti la Sig.na Manelli."#, r#"Arrivederci."#],
    },
    Case {
        text: r#"La centrale meteor. si è guastata. Gli idraul. son dovuti andare a sistemarla."#,
        expected: &[r#"La centrale meteor. si è guastata."#, r#"Gli idraul. son dovuti andare a sistemarla."#],
    },
    Case {
        text: r#"Hanno creato un algoritmo allo st. d. arte. Si ringrazia lo psicol. Serenti."#,
        expected: &[r#"Hanno creato un algoritmo allo st. d. arte."#, r#"Si ringrazia lo psicol. Serenti."#],
    },
    Case {
        text: r#"Chiamate il V.Cte. delle F.P., adesso!"#,
        expected: &[r#"Chiamate il V.Cte. delle F.P., adesso!"#],
    },
    Case {
        text: r#"Giancarlo ha sostenuto l'esame di econ. az.."#,
        expected: &[r#"Giancarlo ha sostenuto l'esame di econ. az.."#],
    },
    Case {
        text: r#"Stava viaggiando a 90 km/h verso la provincia di TR quando il Dott. Mesini ha sentito un rumore e si fermò!"#,
        expected: &[r#"Stava viaggiando a 90 km/h verso la provincia di TR quando il Dott. Mesini ha sentito un rumore e si fermò!"#],
    },
    Case {
        text: r#"Egregio Dir. Amm., le faccio sapere che l'ascensore non funziona."#,
        expected: &[r#"Egregio Dir. Amm., le faccio sapere che l'ascensore non funziona."#],
    },
    Case {
        text: r#"Stava mangiando e/o dormendo."#,
        expected: &[r#"Stava mangiando e/o dormendo."#],
    },
    Case {
        text: r#"Ricordatevi che dom 25 Set. sarà il compleanno di Maria; dovremo darle un regalo."#,
        expected: &[r#"Ricordatevi che dom 25 Set. sarà il compleanno di Maria; dovremo darle un regalo."#],
    },
    Case {
        text: r#"La politica è quella della austerità; quindi verranno fatti tagli agli sprechi."#,
        expected: &[r#"La politica è quella della austerità; quindi verranno fatti tagli agli sprechi."#],
    },
    Case {
        text: r#"Nel tribunale, l'Avv. Fabrizi ha urlato "Io, l'illustrissimo Fabrizi, vi si oppone!"."#,
        expected: &[r#"Nel tribunale, l'Avv. Fabrizi ha urlato "Io, l'illustrissimo Fabrizi, vi si oppone!"."#],
    },
    Case {
        text: r#"Le parti fisiche di un computer (ad es. RAM, CPU, tastiera, mouse, etc.) sono definiti HW."#,
        expected: &[r#"Le parti fisiche di un computer (ad es. RAM, CPU, tastiera, mouse, etc.) sono definiti HW."#],
    },
    Case {
        text: r#"La parola 'casa' è sinonimo di abitazione."#,
        expected: &[r#"La parola 'casa' è sinonimo di abitazione."#],
    },
    Case {
        text: r#"La "Mulino Bianco" fa alimentari pre-confezionati."#,
        expected: &[r#"La "Mulino Bianco" fa alimentari pre-confezionati."#],
    },
    Case {
        text: r#""Ei fu. Siccome immobile / dato il mortal sospiro / stette la spoglia immemore / orba di tanto spiro / [...]" (Manzoni)."#,
        expected: &[r#""Ei fu. Siccome immobile / dato il mortal sospiro / stette la spoglia immemore / orba di tanto spiro / [...]" (Manzoni)."#],
    },
    Case {
        text: r#"Una lettera si può iniziare in questo modo «Il/la sottoscritto/a ... nato/a a ...»."#,
        expected: &[r#"Una lettera si può iniziare in questo modo «Il/la sottoscritto/a ... nato/a a ...»."#],
    },
    Case {
        text: r#"Per casa, in uno degli esercizi per i bambini c'era "3 + (14/7) = 5""#,
        expected: &[r#"Per casa, in uno degli esercizi per i bambini c'era "3 + (14/7) = 5""#],
    },
    Case {
        text: r#"Ai bambini è stato chiesto di fare "4:2*2""#,
        expected: &[r#"Ai bambini è stato chiesto di fare "4:2*2""#],
    },
    Case {
        text: r#"La maestra esclamò: "Bambini, quanto fa '2/3 + 4/3?'"."#,
        expected: &[r#"La maestra esclamò: "Bambini, quanto fa '2/3 + 4/3?'"."#],
    },
    Case {
        text: r#"Il motore misurava 120°C."#,
        expected: &[r#"Il motore misurava 120°C."#],
    },
    Case {
        text: r#"Il volume era di 3m³."#,
        expected: &[r#"Il volume era di 3m³."#],
    },
    Case {
        text: r#"La stanza misurava 20m²."#,
        expected: &[r#"La stanza misurava 20m²."#],
    },
    Case {
        text: r#"1°C corrisponde a 33.8°F."#,
        expected: &[r#"1°C corrisponde a 33.8°F."#],
    },
    Case {
        text: r#"Oggi è il 27-10-14."#,
        expected: &[r#"Oggi è il 27-10-14."#],
    },
    Case {
        text: r#"La casa costa 170.500.000,00€!"#,
        expected: &[r#"La casa costa 170.500.000,00€!"#],
    },
    Case {
        text: r#"Il corridore 103 è arrivato 4°."#,
        expected: &[r#"Il corridore 103 è arrivato 4°."#],
    },
    Case {
        text: r#"Oggi è il 27/10/2014."#,
        expected: &[r#"Oggi è il 27/10/2014."#],
    },
    Case {
        text: r#"Ecco l'elenco: 1.gelato, 2.carne, 3.riso."#,
        expected: &[r#"Ecco l'elenco: 1.gelato, 2.carne, 3.riso."#],
    },
    Case {
        text: r#"Devi comprare : 1)pesce 2)sale."#,
        expected: &[r#"Devi comprare : 1)pesce 2)sale."#],
    },
    Case {
        text: r#"La macchina viaggiava a 100 km/h."#,
        expected: &[r#"La macchina viaggiava a 100 km/h."#],
    },
];

#[test]
fn upstream_it_more_test_cases() {
    assert_cases(IT_MORE_TEST_CASES, Options { clean: false, doc_type: None });
}
