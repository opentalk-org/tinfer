use super::merge_timed;
use crate::{Alignment, AlignmentItem, AlignmentType, AudioChunk};

#[test]
fn full_response_timestamps_follow_the_concatenated_audio() {
    let chunks = [3_200, 950]
        .into_iter()
        .enumerate()
        .map(|(index, duration)| AudioChunk {
            audio: vec![0.1; duration],
            sample_rate: 1_000,
            chunk_index: index as u64,
            text_span: 0..4,
            alignment: Some(Alignment {
                kind: AlignmentType::Char,
                items: vec![AlignmentItem {
                    item: "a".into(),
                    char_start: 0,
                    char_end: 1,
                    start_ms: 0,
                    end_ms: duration as u64,
                }],
            }),
        })
        .collect();

    let merged = merge_timed(chunks).unwrap_or_else(|_| panic!("valid audio chunks must merge"));
    let items = merged.alignment.unwrap().items;
    assert_eq!((items[0].start_ms, items[0].end_ms), (0, 3_200));
    assert_eq!((items[1].start_ms, items[1].end_ms), (3_200, 4_150));
    assert_eq!(merged.audio.len(), 4_150);
}
