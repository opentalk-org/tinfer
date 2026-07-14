use pysbd::{Options, Segmenter};

mod part_01;
mod part_02;
mod part_03;
mod part_04;
mod part_05;

struct Case {
    text: &'static str,
    expected: &'static [&'static str],
}

#[test]
fn english_clean_rows_match_upstream() {
    let groups = [part_01::CASES, part_02::CASES, part_03::CASES, part_04::CASES];
    assert_rows(true, &groups);
}

#[test]
fn english_raw_rows_match_upstream() {
    assert_rows(false, &[part_05::CASES]);
}

fn assert_rows(clean: bool, groups: &[&[Case]]) {
    let segmenter = Segmenter::new("en", Options { clean, doc_type: None }).unwrap();
    for (index, case) in groups.iter().flat_map(|group| group.iter()).enumerate() {
        let actual = segmenter.segment(case.text).unwrap();
        let trimmed: Vec<&str> = actual.iter().map(|sentence| sentence.trim()).collect();
        assert_eq!(trimmed, case.expected, "English clean={clean} row {index}");
    }
}
