use pysbd::{DocType, Options, Segmenter};

struct Case {
    text: &'static str,
    expected: &'static [&'static str],
}

fn assert_cases(cases: &[Case], options: Options) {
    let segmenter = Segmenter::new("de", options).unwrap();
    for case in cases {
        let actual = segmenter.segment(case.text).unwrap();
        let actual: Vec<&str> = actual.iter().map(|sentence| sentence.trim()).collect();
        assert_eq!(actual, case.expected, "input: {}", case.text);
    }
}

const GOLDEN_DE_RULES_TEST_CASES: &[Case] = &[
    Case {
        text: r#"„Ich habe heute keine Zeit“, sagte die Frau und flüsterte leise: „Und auch keine Lust.“ Wir haben 1.000.000 Euro."#,
        expected: &[r#"„Ich habe heute keine Zeit“, sagte die Frau und flüsterte leise: „Und auch keine Lust.“"#, r#"Wir haben 1.000.000 Euro."#],
    },
    Case {
        text: r#"Es gibt jedoch einige Vorsichtsmaßnahmen, die Du ergreifen kannst, z. B. ist es sehr empfehlenswert, dass Du Dein Zuhause von allem Junkfood befreist."#,
        expected: &[r#"Es gibt jedoch einige Vorsichtsmaßnahmen, die Du ergreifen kannst, z. B. ist es sehr empfehlenswert, dass Du Dein Zuhause von allem Junkfood befreist."#],
    },
    Case {
        text: r#"Was sind die Konsequenzen der Abstimmung vom 12. Juni?"#,
        expected: &[r#"Was sind die Konsequenzen der Abstimmung vom 12. Juni?"#],
    },
];

#[test]
fn upstream_golden_de_rules_test_cases() {
    assert_cases(GOLDEN_DE_RULES_TEST_CASES, Options { clean: false, doc_type: None });
}

const DE_CLEAN_RULES_TEST_CASES: &[Case] = &[
    Case {
        text: r#"„Ich habe heute keine Zeit“, sagte die Frau und flüsterte leise: „Und auch keine Lust.“ Wir haben 1.000.000 Euro."#,
        expected: &[r#"„Ich habe heute keine Zeit“, sagte die Frau und flüsterte leise: „Und auch keine Lust.“"#, r#"Wir haben 1.000.000 Euro."#],
    },
    Case {
        text: r#"Thomas sagte: ,,Wann kommst zu mir?” ,,Das weiß ich noch nicht“, antwortete Susi, ,,wahrscheinlich am Sonntag.“ Wir haben 1.000.000 Euro."#,
        expected: &[r#"Thomas sagte: ,,Wann kommst zu mir?” ,,Das weiß ich noch nicht“, antwortete Susi, ,,wahrscheinlich am Sonntag.“"#, r#"Wir haben 1.000.000 Euro."#],
    },
    Case {
        text: r#"„Lass uns jetzt essen gehen!“, sagte die Mutter zu ihrer Freundin, „am besten zum Italiener.“"#,
        expected: &[r#"„Lass uns jetzt essen gehen!“, sagte die Mutter zu ihrer Freundin, „am besten zum Italiener.“"#],
    },
    Case {
        text: r#"Wir haben 1.000.000 Euro."#,
        expected: &[r#"Wir haben 1.000.000 Euro."#],
    },
    Case {
        text: r#"Sie bekommen 3,50 Euro zurück."#,
        expected: &[r#"Sie bekommen 3,50 Euro zurück."#],
    },
    Case {
        text: r#"Dafür brauchen wir 5,5 Stunden."#,
        expected: &[r#"Dafür brauchen wir 5,5 Stunden."#],
    },
    Case {
        text: r#"Bitte überweisen Sie 5.300,25 Euro."#,
        expected: &[r#"Bitte überweisen Sie 5.300,25 Euro."#],
    },
    Case {
        text: r#"1. Dies ist eine Punkteliste."#,
        expected: &[r#"1. Dies ist eine Punkteliste."#],
    },
    Case {
        text: r#"Wir trafen Dr. med. Meyer in der Stadt."#,
        expected: &[r#"Wir trafen Dr. med. Meyer in der Stadt."#],
    },
    Case {
        text: r#"Wir brauchen Getränke, z. B. Wasser, Saft, Bier usw."#,
        expected: &[r#"Wir brauchen Getränke, z. B. Wasser, Saft, Bier usw."#],
    },
    Case {
        text: r#"Ich kann u.a. Spanisch sprechen."#,
        expected: &[r#"Ich kann u.a. Spanisch sprechen."#],
    },
    Case {
        text: r#"Frau Prof. Schulze ist z. Z. nicht da."#,
        expected: &[r#"Frau Prof. Schulze ist z. Z. nicht da."#],
    },
    Case {
        text: r#"Sie erhalten ein neues Bank-Statement bzw. ein neues Schreiben."#,
        expected: &[r#"Sie erhalten ein neues Bank-Statement bzw. ein neues Schreiben."#],
    },
    Case {
        text: r#"Z. T. ist die Lieferung unvollständig."#,
        expected: &[r#"Z. T. ist die Lieferung unvollständig."#],
    },
    Case {
        text: r#"Das finden Sie auf S. 225."#,
        expected: &[r#"Das finden Sie auf S. 225."#],
    },
    Case {
        text: r#"Sie besucht eine kath. Schule."#,
        expected: &[r#"Sie besucht eine kath. Schule."#],
    },
    Case {
        text: r#"Wir benötigen Zeitungen, Zeitschriften u. Ä. für unser Projekt."#,
        expected: &[r#"Wir benötigen Zeitungen, Zeitschriften u. Ä. für unser Projekt."#],
    },
    Case {
        text: r#"Das steht auf S. 23, s. vorherige Anmerkung."#,
        expected: &[r#"Das steht auf S. 23, s. vorherige Anmerkung."#],
    },
    Case {
        text: r#"Dies ist meine Adresse: Dr. Meier, Berliner Str. 5, 21234 Bremen."#,
        expected: &[r#"Dies ist meine Adresse: Dr. Meier, Berliner Str. 5, 21234 Bremen."#],
    },
    Case {
        text: r#"Er sagte: „Hallo, wie geht´s Ihnen, Frau Prof. Müller?“"#,
        expected: &[r#"Er sagte: „Hallo, wie geht´s Ihnen, Frau Prof. Müller?“"#],
    },
    Case {
        text: r#"Fit in vier Wochen

Deine Anleitung für eine reine Ernährung und ein gesünderes und glücklicheres Leben

RECHTLICHE HINWEISE

Ohne die ausdrückliche schriftliche Genehmigung der Eigentümerin von instafemmefitness, Anna Anderson, darf dieses E-Book weder teilweise noch in vollem Umfang reproduziert, gespeichert, kopiert oder auf irgendeine Weise übertragen werden. Wenn Du das E-Book auf einem öffentlich zugänglichen Computer ausdruckst, musst Du es nach dem Ausdrucken von dem Computer löschen. Jedes E-Book wird mit einem Benutzernamen und Transaktionsinformationen versehen.

Verstöße gegen dieses Urheberrecht werden im vollen gesetzlichen Umfang geltend gemacht. Obgleich die Autorin und Herausgeberin alle Anstrengungen unternommen hat, sicherzustellen, dass die Informationen in diesem Buch zum Zeitpunkt der Drucklegung korrekt sind, übernimmt die Autorin und Herausgeberin keine Haftung für etwaige Verluste, Schäden oder Störungen, die durch Fehler oder Auslassungen in Folge von Fahrlässigkeit, zufälligen Umständen oder sonstigen Ursachen entstehen, und lehnt hiermit jedwede solche Haftung ab.

Dieses Buch ist kein Ersatz für die medizinische Beratung durch Ärzte. Der Leser/die Leserin sollte regelmäßig einen Arzt/eine Ärztin hinsichtlich Fragen zu seiner/ihrer Gesundheit und vor allem in Bezug auf Symptome, die eventuell einer ärztlichen Diagnose oder Behandlung bedürfen, konsultieren.

Die Informationen in diesem Buch sind dazu gedacht, ein ordnungsgemäßes Training zu ergänzen, nicht aber zu ersetzen. Wie jeder andere Sport, der Geschwindigkeit, Ausrüstung, Gleichgewicht und Umweltfaktoren einbezieht, stellt dieser Sport ein gewisses Risiko dar. Die Autorin und Herausgeberin rät den Lesern dazu, die volle Verantwortung für die eigene Sicherheit zu übernehmen und die eigenen Grenzen zu beachten. Vor dem Ausüben der in diesem Buch beschriebenen Übungen solltest Du sicherstellen, dass Deine Ausrüstung in gutem Zustand ist, und Du solltest keine Risiken außerhalb Deines Erfahrungs- oder Trainingsniveaus, Deiner Fähigkeiten oder Deines Komfortbereichs eingehen.
Hintergrundillustrationen Urheberrecht © 2013 bei Shuttershock, Buchgestaltung und -produktion durch Anna Anderson Verfasst von Anna Anderson
Urheberrecht © 2014 Instafemmefitness. Alle Rechte vorbehalten

Über mich"#,
        expected: &[r#"Fit in vier Wochen"#, r#"Deine Anleitung für eine reine Ernährung und ein gesünderes und glücklicheres Leben"#, r#"RECHTLICHE HINWEISE"#, r#"Ohne die ausdrückliche schriftliche Genehmigung der Eigentümerin von instafemmefitness, Anna Anderson, darf dieses E-Book weder teilweise noch in vollem Umfang reproduziert, gespeichert, kopiert oder auf irgendeine Weise übertragen werden."#, r#"Wenn Du das E-Book auf einem öffentlich zugänglichen Computer ausdruckst, musst Du es nach dem Ausdrucken von dem Computer löschen."#, r#"Jedes E-Book wird mit einem Benutzernamen und Transaktionsinformationen versehen."#, r#"Verstöße gegen dieses Urheberrecht werden im vollen gesetzlichen Umfang geltend gemacht."#, r#"Obgleich die Autorin und Herausgeberin alle Anstrengungen unternommen hat, sicherzustellen, dass die Informationen in diesem Buch zum Zeitpunkt der Drucklegung korrekt sind, übernimmt die Autorin und Herausgeberin keine Haftung für etwaige Verluste, Schäden oder Störungen, die durch Fehler oder Auslassungen in Folge von Fahrlässigkeit, zufälligen Umständen oder sonstigen Ursachen entstehen, und lehnt hiermit jedwede solche Haftung ab."#, r#"Dieses Buch ist kein Ersatz für die medizinische Beratung durch Ärzte."#, r#"Der Leser/die Leserin sollte regelmäßig einen Arzt/eine Ärztin hinsichtlich Fragen zu seiner/ihrer Gesundheit und vor allem in Bezug auf Symptome, die eventuell einer ärztlichen Diagnose oder Behandlung bedürfen, konsultieren."#, r#"Die Informationen in diesem Buch sind dazu gedacht, ein ordnungsgemäßes Training zu ergänzen, nicht aber zu ersetzen."#, r#"Wie jeder andere Sport, der Geschwindigkeit, Ausrüstung, Gleichgewicht und Umweltfaktoren einbezieht, stellt dieser Sport ein gewisses Risiko dar."#, r#"Die Autorin und Herausgeberin rät den Lesern dazu, die volle Verantwortung für die eigene Sicherheit zu übernehmen und die eigenen Grenzen zu beachten."#, r#"Vor dem Ausüben der in diesem Buch beschriebenen Übungen solltest Du sicherstellen, dass Deine Ausrüstung in gutem Zustand ist, und Du solltest keine Risiken außerhalb Deines Erfahrungs- oder Trainingsniveaus, Deiner Fähigkeiten oder Deines Komfortbereichs eingehen."#, r#"Hintergrundillustrationen Urheberrecht © 2013 bei Shuttershock, Buchgestaltung und -produktion durch Anna Anderson Verfasst von Anna Anderson"#, r#"Urheberrecht © 2014 Instafemmefitness."#, r#"Alle Rechte vorbehalten"#, r#"Über mich"#],
    },
    Case {
        text: r#"Es gibt jedoch einige Vorsichtsmaßnahmen, die Du ergreifen kannst, z. B. ist es sehr empfehlenswert, dass Du Dein Zuhause von allem Junkfood befreist. Ich persönlich kaufe kein Junkfood oder etwas, das nicht rein ist (ich traue mir da selbst nicht!). Ich finde jeden Vorwand, um das Junkfood zu essen, vor allem die Vorstellung, dass ich nicht mehr in Versuchung kommen werde, wenn ich es jetzt aufesse und es weg ist. Es ist schon komisch, was unser Verstand mitunter anstellt!"#,
        expected: &[r#"Es gibt jedoch einige Vorsichtsmaßnahmen, die Du ergreifen kannst, z. B. ist es sehr empfehlenswert, dass Du Dein Zuhause von allem Junkfood befreist."#, r#"Ich persönlich kaufe kein Junkfood oder etwas, das nicht rein ist (ich traue mir da selbst nicht!)."#, r#"Ich finde jeden Vorwand, um das Junkfood zu essen, vor allem die Vorstellung, dass ich nicht mehr in Versuchung kommen werde, wenn ich es jetzt aufesse und es weg ist."#, r#"Es ist schon komisch, was unser Verstand mitunter anstellt!"#],
    },
    Case {
        text: r#"Ob Sie in Hannover nur auf der Durchreise, für einen längeren Aufenthalt oder zum Besuch einer der zahlreichen Messen sind: Die Hauptstadt des Landes Niedersachsens hat viele Sehenswürdigkeiten und ist zu jeder Jahreszeit eine Reise Wert. 
Hannovers Ursprünge können bis zur römischen Kaiserzeit zurückverfolgt werden, und zwar durch Ausgrabungen von Tongefäßen aus dem 1. -3. Jahrhundert nach Christus, die an mehreren Stellen im Untergrund des Stadtzentrums durchgeführt wurden."#,
        expected: &[r#"Ob Sie in Hannover nur auf der Durchreise, für einen längeren Aufenthalt oder zum Besuch einer der zahlreichen Messen sind: Die Hauptstadt des Landes Niedersachsens hat viele Sehenswürdigkeiten und ist zu jeder Jahreszeit eine Reise Wert."#, r#"Hannovers Ursprünge können bis zur römischen Kaiserzeit zurückverfolgt werden, und zwar durch Ausgrabungen von Tongefäßen aus dem 1. -3. Jahrhundert nach Christus, die an mehreren Stellen im Untergrund des Stadtzentrums durchgeführt wurden."#],
    },
    Case {
        text: r#"• 3. Seien Sie achtsam bei der Auswahl der Nahrungsmittel! 
• 4. Nehmen Sie zusätzlich Folsäurepräparate und essen Sie Fisch! 
• 5. Treiben Sie regelmäßig Sport! 
• 6. Beginnen Sie mit Übungen für die Beckenbodenmuskulatur! 
• 7. Reduzieren Sie Ihren Alkoholgenuss! 
"#,
        expected: &[r#"• 3. Seien Sie achtsam bei der Auswahl der Nahrungsmittel!"#, r#"• 4. Nehmen Sie zusätzlich Folsäurepräparate und essen Sie Fisch!"#, r#"• 5. Treiben Sie regelmäßig Sport!"#, r#"• 6. Beginnen Sie mit Übungen für die Beckenbodenmuskulatur!"#, r#"• 7. Reduzieren Sie Ihren Alkoholgenuss!"#],
    },
    Case {
        text: r#"Was sind die Konsequenzen der Abstimmung vom 12. Juni?"#,
        expected: &[r#"Was sind die Konsequenzen der Abstimmung vom 12. Juni?"#],
    },
    Case {
        text: r#"Was pro Jahr10. Zudem pro Jahr um 0.3 %11. Der gängigen Theorie nach erfolgt der Anstieg."#,
        expected: &[r#"Was pro Jahr10."#, r#"Zudem pro Jahr um 0.3 %11."#, r#"Der gängigen Theorie nach erfolgt der Anstieg."#],
    },
    Case {
        text: r#"s. vorherige Anmerkung."#,
        expected: &[r#"s. vorherige Anmerkung."#],
    },
    Case {
        text: r#"Mit Inkrafttreten des Mindestlohngesetzes (MiLoG) zum 01. Januar 2015 werden in Bezug auf den Einsatz von Leistungs."#,
        expected: &[r#"Mit Inkrafttreten des Mindestlohngesetzes (MiLoG) zum 01. Januar 2015 werden in Bezug auf den Einsatz von Leistungs."#],
    },
    Case {
        text: r#"
• einige Sorten Weichkäse  
• rohes oder nicht ganz durchgebratenes Fleisch  
• ungeputztes Gemüse und ungewaschener Salat  
• nicht ganz durchgebratenes Hühnerfleisch, rohe oder nur weich gekochte Eier"#,
        expected: &[r#"• einige Sorten Weichkäse"#, r#"• rohes oder nicht ganz durchgebratenes Fleisch"#, r#"• ungeputztes Gemüse und ungewaschener Salat"#, r#"• nicht ganz durchgebratenes Hühnerfleisch, rohe oder nur weich gekochte Eier"#],
    },
];

#[test]
fn upstream_de_clean_rules_test_cases() {
    assert_cases(DE_CLEAN_RULES_TEST_CASES, Options { clean: true, doc_type: None });
}

const DE_PDF_CLEAN_RULES_TEST_CASES: &[Case] = &[
    Case {
        text: r#"
   

   http:www.babycentre.co.uk/midwives 

 

 

10 steps to a healthy pregnancy (German) 

10 Schritte zu einer gesunden Schwangerschaft 
 
• 1. Planen und organisieren Sie die Zeit der Schwangerschaft frühzeitig! 
• 2. Essen Sie gesund! 
• 3. Seien Sie achtsam bei der Auswahl der Nahrungsmittel! 
• 4. Nehmen Sie zusätzlich Folsäurepräparate und essen Sie Fisch! 
• 5. Treiben Sie regelmäßig Sport! 
• 6. Beginnen Sie mit Übungen für die Beckenbodenmuskulatur! 
• 7. Reduzieren Sie Ihren Alkoholgenuss! 
• 8. Reduzieren Sie Ihren Koffeingenuß! 
• 9. Hören Sie mit dem Rauchen auf! 
• 10. Gönnen Sie sich Erholung! 
 
 
Zehn einfach zu befolgende Tipps sollen Ihnen helfen, eine möglichst problemlose 
Schwangerschaft zu erleben und ein gesundes Baby auf die Welt zu bringen:  

1. Planen und organisieren Sie die Zeit der Schwangerschaft frühzeitig!"#,
        expected: &[r#"http:www.babycentre.co.uk/midwives"#, r#"10 steps to a healthy pregnancy (German)"#, r#"10 Schritte zu einer gesunden Schwangerschaft"#, r#"• 1. Planen und organisieren Sie die Zeit der Schwangerschaft frühzeitig!"#, r#"• 2. Essen Sie gesund!"#, r#"• 3. Seien Sie achtsam bei der Auswahl der Nahrungsmittel!"#, r#"• 4. Nehmen Sie zusätzlich Folsäurepräparate und essen Sie Fisch!"#, r#"• 5. Treiben Sie regelmäßig Sport!"#, r#"• 6. Beginnen Sie mit Übungen für die Beckenbodenmuskulatur!"#, r#"• 7. Reduzieren Sie Ihren Alkoholgenuss!"#, r#"• 8. Reduzieren Sie Ihren Koffeingenuß!"#, r#"• 9. Hören Sie mit dem Rauchen auf!"#, r#"• 10. Gönnen Sie sich Erholung!"#, r#"Zehn einfach zu befolgende Tipps sollen Ihnen helfen, eine möglichst problemlose Schwangerschaft zu erleben und ein gesundes Baby auf die Welt zu bringen:"#, r#"1. Planen und organisieren Sie die Zeit der Schwangerschaft frühzeitig!"#],
    },
    Case {
        text: r#"Schwangere Frauen sollten während der 
ersten drei Monate eine tägliche Dosis von 400 Mikrogramm Folsäure zusätzlich nehmen. 
Folsäure befindet sich auch in einigen Gemüse- und Müslisorten."#,
        expected: &[r#"Schwangere Frauen sollten während der ersten drei Monate eine tägliche Dosis von 400 Mikrogramm Folsäure zusätzlich nehmen."#, r#"Folsäure befindet sich auch in einigen Gemüse- und Müslisorten."#],
    },
    Case {
        text: r#"Andere 
Fischsorten (z.B. Hai, Thunfisch, Aal und Seeteufel) weisen einen erhöhten Quecksilbergehalt 
auf und sollten deshalb in der Schwangerschaft nur selten verzehrt werden."#,
        expected: &[r#"Andere Fischsorten (z. B. Hai, Thunfisch, Aal und Seeteufel) weisen einen erhöhten Quecksilbergehalt auf und sollten deshalb in der Schwangerschaft nur selten verzehrt werden."#],
    },
];

#[test]
fn upstream_de_pdf_clean_rules_test_cases() {
    assert_cases(DE_PDF_CLEAN_RULES_TEST_CASES, Options { clean: true, doc_type: Some(DocType::Pdf) });
}
