use pysbd::{DocType, Options, Segmenter};

struct Case {
    text: &'static str,
    expected: &'static [&'static str],
}

#[rustfmt::skip]
const GOLDEN_ES_RULES_TEST_CASES: &[Case] = &[
    Case { text: "¿Cómo está hoy? Espero que muy bien.", expected: &["¿Cómo está hoy?", "Espero que muy bien."] },
    Case { text: "¡Hola señorita! Espero que muy bien.", expected: &["¡Hola señorita!", "Espero que muy bien."] },
    Case { text: "Hola Srta. Ledesma. Buenos días, soy el Lic. Naser Pastoriza, y él es mi padre, el Dr. Naser.", expected: &["Hola Srta. Ledesma.", "Buenos días, soy el Lic. Naser Pastoriza, y él es mi padre, el Dr. Naser."] },
    Case { text: "¡La casa cuesta $170.500.000,00! ¡Muy costosa! Se prevé una disminución del 12.5% para el próximo año.", expected: &["¡La casa cuesta $170.500.000,00!", "¡Muy costosa!", "Se prevé una disminución del 12.5% para el próximo año."] },
    Case { text: "«Ninguna mente extraordinaria está exenta de un toque de demencia.», dijo Aristóteles.", expected: &["«Ninguna mente extraordinaria está exenta de un toque de demencia.», dijo Aristóteles."] },
];

#[rustfmt::skip]
const ES_MORE_TEST_CASES: &[Case] = &[
    Case { text: "«Ninguna mente extraordinaria está exenta de un toque de demencia», dijo Aristóteles. Pablo, ¿adónde vas? ¡¿Qué viste?!", expected: &["«Ninguna mente extraordinaria está exenta de un toque de demencia», dijo Aristóteles.", "Pablo, ¿adónde vas?", "¡¿Qué viste?!"] },
    Case { text: "Admón. es administración o me equivoco.", expected: &["Admón. es administración o me equivoco."] },
    Case { text: "¡Hola Srta. Ledesma! ¿Cómo está hoy? Espero que muy bien.", expected: &["¡Hola Srta. Ledesma!", "¿Cómo está hoy?", "Espero que muy bien."] },
    Case { text: "Buenos días, soy el Lic. Naser Pastoriza, y él es mi padre, el Dr. Naser.", expected: &["Buenos días, soy el Lic. Naser Pastoriza, y él es mi padre, el Dr. Naser."] },
    Case { text: "He apuntado una cita para la siguiente fecha: Mar. 23 de Nov. de 2014. Gracias.", expected: &["He apuntado una cita para la siguiente fecha: Mar. 23 de Nov. de 2014.", "Gracias."] },
    Case { text: "Núm. de tel: 351.123.465.4. Envíe mis saludos a la Sra. Rescia.", expected: &["Núm. de tel: 351.123.465.4.", "Envíe mis saludos a la Sra. Rescia."] },
    Case { text: "Cero en la escala Celsius o de grados centígrados (0 °C) se define como el equivalente a 273.15 K, con una diferencia de temperatura de 1 °C equivalente a una diferencia de 1 Kelvin. Esto significa que 100 °C, definido como el punto de ebullición del agua, se define como el equivalente a 373.15 K.", expected: &["Cero en la escala Celsius o de grados centígrados (0 °C) se define como el equivalente a 273.15 K, con una diferencia de temperatura de 1 °C equivalente a una diferencia de 1 Kelvin.", "Esto significa que 100 °C, definido como el punto de ebullición del agua, se define como el equivalente a 373.15 K."] },
    Case { text: "Durante la primera misión del Discovery (30 Ago. 1984 15:08.10) tuvo lugar el lanzamiento de dos satélites de comunicación, el nombre de esta misión fue STS-41-D.", expected: &["Durante la primera misión del Discovery (30 Ago. 1984 15:08.10) tuvo lugar el lanzamiento de dos satélites de comunicación, el nombre de esta misión fue STS-41-D."] },
    Case { text: "Frase del gran José Hernández: \"Aquí me pongo a cantar / al compás de la vigüela, / que el hombre que lo desvela / una pena estrordinaria, / como la ave solitaria / con el cantar se consuela. / [...] \".", expected: &["Frase del gran José Hernández: \"Aquí me pongo a cantar / al compás de la vigüela, / que el hombre que lo desvela / una pena estrordinaria, / como la ave solitaria / con el cantar se consuela. / [...] \"."] },
    Case { text: "Citando a Criss Jami «Prefiero ser un artista a ser un líder, irónicamente, un líder tiene que seguir las reglas.», lo cual parece muy acertado.", expected: &["Citando a Criss Jami «Prefiero ser un artista a ser un líder, irónicamente, un líder tiene que seguir las reglas.», lo cual parece muy acertado."] },
    Case { text: "Cuando llegué, le estaba dando ejercicios a los niños, uno de los cuales era \"3 + (14/7).x = 5\". ¿Qué te parece?", expected: &["Cuando llegué, le estaba dando ejercicios a los niños, uno de los cuales era \"3 + (14/7).x = 5\".", "¿Qué te parece?"] },
    Case { text: "Se le pidió a los niños que leyeran los párrf. 5 y 6 del art. 4 de la constitución de los EE. UU..", expected: &["Se le pidió a los niños que leyeran los párrf. 5 y 6 del art. 4 de la constitución de los EE. UU.."] },
    Case { text: "Una de las preguntas realizadas en la evaluación del día Lun. 15 de Mar. fue la siguiente: \"Alumnos, ¿cuál es el resultado de la operación 1.1 + 4/5?\". Disponían de 1 min. para responder esa pregunta.", expected: &["Una de las preguntas realizadas en la evaluación del día Lun. 15 de Mar. fue la siguiente: \"Alumnos, ¿cuál es el resultado de la operación 1.1 + 4/5?\".", "Disponían de 1 min. para responder esa pregunta."] },
    Case { text: "La temperatura del motor alcanzó los 120.5°C. Afortunadamente, pudo llegar al final de carrera.", expected: &["La temperatura del motor alcanzó los 120.5°C.", "Afortunadamente, pudo llegar al final de carrera."] },
    Case { text: "El volumen del cuerpo es 3m³. ¿Cuál es la superficie de cada cara del prisma?", expected: &["El volumen del cuerpo es 3m³.", "¿Cuál es la superficie de cada cara del prisma?"] },
    Case { text: "La habitación tiene 20.55m². El living tiene 50.0m².", expected: &["La habitación tiene 20.55m².", "El living tiene 50.0m²."] },
    Case { text: "1°C corresponde a 33.8°F. ¿A cuánto corresponde 35°C?", expected: &["1°C corresponde a 33.8°F.", "¿A cuánto corresponde 35°C?"] },
    Case { text: "Hamilton ganó el último gran premio de Fórmula 1, luego de 1:39:02.619 Hs. de carrera, segundo resultó Massa, a una diferencia de 2.5 segundos. De esta manera se consagró ¡Campeón mundial!", expected: &["Hamilton ganó el último gran premio de Fórmula 1, luego de 1:39:02.619 Hs. de carrera, segundo resultó Massa, a una diferencia de 2.5 segundos.", "De esta manera se consagró ¡Campeón mundial!"] },
    Case { text: "¡La casa cuesta $170.500.000,00! ¡Muy costosa! Se prevé una disminución del 12.5% para el próximo año.", expected: &["¡La casa cuesta $170.500.000,00!", "¡Muy costosa!", "Se prevé una disminución del 12.5% para el próximo año."] },
    Case { text: "El corredor No. 103 arrivó 4°.", expected: &["El corredor No. 103 arrivó 4°."] },
    Case { text: "Hoy es 27/04/2014, y es mi cumpleaños. ¿Cuándo es el tuyo?", expected: &["Hoy es 27/04/2014, y es mi cumpleaños.", "¿Cuándo es el tuyo?"] },
    Case { text: "Aquí está la lista de compras para el almuerzo: 1.Helado, 2.Carne, 3.Arroz. ¿Cuánto costará? Quizás $12.5.", expected: &["Aquí está la lista de compras para el almuerzo: 1.Helado, 2.Carne, 3.Arroz.", "¿Cuánto costará?", "Quizás $12.5."] },
    Case { text: "1 + 1 es 2. 2 + 2 es 4. El auto es de color rojo.", expected: &["1 + 1 es 2.", "2 + 2 es 4.", "El auto es de color rojo."] },
    Case { text: "La máquina viajaba a 100 km/h. ¿En cuánto tiempo recorrió los 153 Km.?", expected: &["La máquina viajaba a 100 km/h.", "¿En cuánto tiempo recorrió los 153 Km.?"] },
    Case { text: "Explora oportunidades de carrera en el área de Salud en el Hospital de Northern en Mt. Kisco.", expected: &["Explora oportunidades de carrera en el área de Salud en el Hospital de Northern en Mt. Kisco."] },
];

#[rustfmt::skip]
const ES_CLEAN_TEST_CASES: &[Case] = &[
    Case { text: "\n \nCentro de Relaciones Interinstitucionales -CERI \n\nCra. 7 No. 40-53 Piso 10 Tel.  (57-1) 3239300 Ext. 1010 Fax: (57-1) 3402973 Bogotá, D.C. - Colombia \n\nhttp://www.udistrital.edu.co - http://ceri.udistrital.edu.co - relinter@udistrital.edu.co \n\n \n\nCERI 0908 \n \nBogotá, D.C. 6 de noviembre de 2014.  \n \nSeñores: \nEMBAJADA DE UNITED KINGDOM \n \n", expected: &["Centro de Relaciones Interinstitucionales -CERI", "Cra. 7 No. 40-53 Piso 10 Tel.  (57-1) 3239300 Ext. 1010 Fax: (57-1) 3402973 Bogotá, D.C. - Colombia", "http://www.udistrital.edu.co - http://ceri.udistrital.edu.co - relinter@udistrital.edu.co", "CERI 0908", "Bogotá, D.C. 6 de noviembre de 2014.", "Señores:", "EMBAJADA DE UNITED KINGDOM"] },
    Case { text: "N°. 1026.253.553", expected: &["N°. 1026.253.553"] },
    Case { text: "\n__________________________________________________________\nEl Board para Servicios Educativos de Putnam/Northern Westchester según el título IX, Sección 504 del “Rehabilitation Act” del 1973, del Título VII y del Acta “American with Disabilities” no discrimina para la admisión a programas educativos por sexo, creencia, nacionalidad, origen, edad o discapacidad.", expected: &["__________________________________________________________", "El Board para Servicios Educativos de Putnam/Northern Westchester según el título IX, Sección 504 del “Rehabilitation Act” del 1973, del Título VII y del Acta “American with Disabilities” no discrimina para la admisión a programas educativos por sexo, creencia, nacionalidad, origen, edad o discapacidad."] },
    Case { text: "• 1. Busca atención prenatal desde el principio \n• 2. Aliméntate bien \n• 3. Presta mucha atención a la higiene de los alimentos \n• 4. Toma suplementos de ácido fólico y come pescado \n• 5. Haz ejercicio regularmente \n• 6. Comienza a hacer ejercicios de Kegel \n• 7. Restringe el consumo de alcohol \n• 8. Disminuye el consumo de cafeína \n• 9. Deja de fumar \n• 10. Descansa", expected: &["• 1. Busca atención prenatal desde el principio", "• 2. Aliméntate bien", "• 3. Presta mucha atención a la higiene de los alimentos", "• 4. Toma suplementos de ácido fólico y come pescado", "• 5. Haz ejercicio regularmente", "• 6. Comienza a hacer ejercicios de Kegel", "• 7. Restringe el consumo de alcohol", "• 8. Disminuye el consumo de cafeína", "• 9. Deja de fumar", "• 10. Descansa"] },
    Case { text: "• 1. Busca atención prenatal desde el principio \n• 2. Aliméntate bien \n• 3. Presta mucha atención a la higiene de los alimentos \n• 4. Toma suplementos de ácido fólico y come pescado \n• 5. Haz ejercicio regularmente \n• 6. Comienza a hacer ejercicios de Kegel \n• 7. Restringe el consumo de alcohol \n• 8. Disminuye el consumo de cafeína \n• 9. Deja de fumar \n• 10. Descansa \n• 11. Hola", expected: &["• 1. Busca atención prenatal desde el principio", "• 2. Aliméntate bien", "• 3. Presta mucha atención a la higiene de los alimentos", "• 4. Toma suplementos de ácido fólico y come pescado", "• 5. Haz ejercicio regularmente", "• 6. Comienza a hacer ejercicios de Kegel", "• 7. Restringe el consumo de alcohol", "• 8. Disminuye el consumo de cafeína", "• 9. Deja de fumar", "• 10. Descansa", "• 11. Hola"] },
];

#[rustfmt::skip]
const ES_PDF_CASE: &[Case] = &[
    Case { text: "\nA continuación me permito presentar a la Ingeniera LAURA MILENA LEÓN \nSANDOVAL, identificada con el documento N°. 1026.253.553 de Bogotá, \negresada del Programa Ingeniería Industrial en el año 2012, quien se desatacó por \nsu excelencia académica, actualmente cursa el programa de Maestría en \nIngeniería Industrial y se encuentra en un intercambio cultural en Bangalore – \nIndia.", expected: &["A continuación me permito presentar a la Ingeniera LAURA MILENA LEÓN SANDOVAL, identificada con el documento N°. 1026.253.553 de Bogotá, egresada del Programa Ingeniería Industrial en el año 2012, quien se desatacó por su excelencia académica, actualmente cursa el programa de Maestría en Ingeniería Industrial y se encuentra en un intercambio cultural en Bangalore – India."] },
];

#[test]
fn golden_rules_match_upstream() {
    let segmenter = Segmenter::new("es", Options { clean: false, doc_type: None }).unwrap();
    for case in GOLDEN_ES_RULES_TEST_CASES {
        let actual = segmenter.segment(case.text).unwrap();
        let trimmed: Vec<&str> = actual.iter().map(|sentence| sentence.trim()).collect();
        assert_eq!(trimmed, case.expected, "{}", case.text);
    }
}

#[test]
fn more_examples_match_upstream() {
    let segmenter = Segmenter::new("es", Options { clean: false, doc_type: None }).unwrap();
    for case in ES_MORE_TEST_CASES {
        let actual = segmenter.segment(case.text).unwrap();
        let trimmed: Vec<&str> = actual.iter().map(|sentence| sentence.trim()).collect();
        assert_eq!(trimmed, case.expected, "{}", case.text);
    }
}

#[test]
fn clean_examples_match_upstream() {
    let segmenter = Segmenter::new("es", Options { clean: true, doc_type: None }).unwrap();
    for case in ES_CLEAN_TEST_CASES {
        let actual = segmenter.segment(case.text).unwrap();
        let trimmed: Vec<&str> = actual.iter().map(|sentence| sentence.trim()).collect();
        assert_eq!(trimmed, case.expected, "{}", case.text);
    }
}

#[test]
fn pdf_example_matches_upstream() {
    let segmenter = Segmenter::new("es", Options { clean: true, doc_type: Some(DocType::Pdf) }).unwrap();
    for case in ES_PDF_CASE {
        let actual = segmenter.segment(case.text).unwrap();
        let trimmed: Vec<&str> = actual.iter().map(|sentence| sentence.trim()).collect();
        assert_eq!(trimmed, case.expected, "{}", case.text);
    }
}
