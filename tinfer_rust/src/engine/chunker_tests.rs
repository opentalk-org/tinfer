use super::chunker::Chunker;

fn texts(chunks: &[super::chunker::PreparedChunk]) -> Vec<&str> {
    chunks.iter().map(|chunk| chunk.text.as_str()).collect()
}

#[test]
fn next_schedule_entry_is_the_no_split_limit() {
    let chunker = Chunker::new("en").unwrap();
    let chunks = chunker.prepare("123456789", 0, 0, &[6, 10]).unwrap();

    assert_eq!(texts(&chunks), ["123456789"]);
    assert_eq!(chunks[0].span, 0..9);
}

#[test]
fn final_limit_is_derived_from_the_previous_delta() {
    let chunker = Chunker::new("en").unwrap();
    let chunks = chunker.prepare("abcdefghijklmnopqrstu", 0, 1, &[4, 7]).unwrap();

    assert_eq!(texts(&chunks), ["abcdefg", "hijklmn", "opqrstu"]);
    assert_eq!(chunks.last().unwrap().span, 14..21);
}

#[test]
fn final_chunks_merge_under_the_derived_no_split_limit() {
    let chunker = Chunker::new("en").unwrap();
    let chunks = chunker.prepare("First sentence. Second sentence. Third.", 0, 0, &[18]).unwrap();

    assert_eq!(texts(&chunks), ["First sentence. ", "Second sentence. Third."]);
    assert_eq!(chunks.iter().map(|chunk| chunk.text.as_str()).collect::<String>(), "First sentence. Second sentence. Third.");
}

#[test]
fn repeated_text_and_trailing_whitespace_keep_source_spans() {
    let chunker = Chunker::new("en").unwrap();
    let chunks = chunker.prepare("go go go   ", 5, 0, &[3]).unwrap();

    assert_eq!(texts(&chunks), ["go ", "go ", "go "]);
    assert_eq!(chunks.iter().map(|chunk| chunk.span.clone()).collect::<Vec<_>>(), [5..8, 8..11, 11..16]);
}

#[test]
fn schedule_lengths_and_spans_count_unicode_characters() {
    let chunker = Chunker::new("en").unwrap();
    let chunks = chunker.prepare("ąęółźćńąęółźćń", 2, 0, &[5]).unwrap();

    assert_eq!(texts(&chunks), ["ąęółź", "ćńąęó", "łźćń"]);
    assert_eq!(chunks.iter().map(|chunk| chunk.span.clone()).collect::<Vec<_>>(), [2..7, 7..12, 12..16]);
}
