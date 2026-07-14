use pysbd::{DocType, Language, clean};

#[test]
fn cleaner_cases_match_upstream() {
    let cases = [
        ("It was a cold \nnight in the city.", "It was a cold night in the city."),
        ("This is the U.S. Senate my friends. <em>Yes.</em> <em>It is</em>!", "This is the U.S. Senate my friends. Yes. It is!"),
    ];

    for (text, expected) in cases {
        let owned = text.to_owned();
        assert_eq!(clean(&owned, Language::English, None).unwrap(), expected);
        assert_eq!(owned, text);
    }
}

#[test]
fn empty_input_is_preserved() {
    assert_eq!(clean("", Language::English, None).unwrap(), "");
}

#[test]
fn pdf_line_breaks_are_joined() {
    assert_eq!(clean("A sentence \ncontinued here.", Language::English, Some(DocType::Pdf)).unwrap(), "A sentence continued here.");
}
