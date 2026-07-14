use pysbd::{DocType, Options, Segmenter};

#[path = "lang/am.rs"]
mod am;
#[path = "lang/ar.rs"]
mod ar;
#[path = "lang/bg.rs"]
mod bg;
#[path = "lang/da.rs"]
mod da;
#[path = "lang/en.rs"]
mod en;
#[path = "english_clean/mod.rs"]
mod english_clean;
#[path = "lang/fr.rs"]
mod fr;
#[path = "lang/hy.rs"]
mod hy;
#[path = "lang/my.rs"]
mod my;
#[path = "lang/nl.rs"]
mod nl;
#[path = "lang/zh.rs"]
mod zh;

fn assert_cases(language: &str, clean: bool, doc_type: Option<DocType>, cases: &[(&str, &[&str])]) {
    let segmenter = Segmenter::new(language, Options { clean, doc_type }).unwrap();
    for (index, (text, expected)) in cases.iter().enumerate() {
        let actual = segmenter.segment(text).unwrap();
        let trimmed: Vec<&str> = actual.iter().map(|sentence| sentence.trim()).collect();
        assert_eq!(trimmed, *expected, "{language} case {index}");
    }
}
