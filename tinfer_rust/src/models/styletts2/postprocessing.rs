use crate::models::base::native::ffi;
use crate::{Alignment, AlignmentItem, AlignmentType, Error, ModelOutput, ModelRequest, Result};

use super::text::{MappedItem, PreparedText};

pub(super) fn finish(
    requests: &[ModelRequest],
    texts: Vec<PreparedText>,
    tensors: Vec<ffi::Tensor>,
    sample_rate: u32,
) -> Result<Vec<ModelOutput>> {
    let audio = find(&tensors, "audio", ffi::DType::F32)?;
    let frames = i32_values(find(&tensors, "frames", ffi::DType::I32)?)?;
    let durations = i32_values(find(&tensors, "durations", ffi::DType::I32)?)?;
    let styles = f32_values(find(&tensors, "style", ffi::DType::F32)?)?;
    let samples = f32_values(audio)?;
    let batch = requests.len();
    let audio_stride = samples.len() / batch;
    let duration_stride = durations.len() / batch;
    requests
        .iter()
        .enumerate()
        .map(|(index, request)| {
            let target = (frames[index] as usize * 600).min(audio_stride).saturating_sub(100);
            let audio_start = index * audio_stride;
            let audio = samples[audio_start..audio_start + target].to_vec();
            let item_durations = &durations[index * duration_stride..(index + 1) * duration_stride];
            let alignment = alignment(request, &texts[index], item_durations);
            let style_start = index * 256;
            let style = styles
                .get(style_start..style_start + 256)
                .ok_or_else(|| Error::Inference("native StyleTTS2 style output differs from batch".into()))?;
            Ok(ModelOutput {
                audio,
                sample_rate,
                alignment,
                state: serde_json::json!({"previous_style_vector": style, "text": request.text}),
            })
        })
        .collect()
}

fn find<'a>(tensors: &'a [ffi::Tensor], name: &str, dtype: ffi::DType) -> Result<&'a ffi::Tensor> {
    tensors
        .iter()
        .find(|tensor| tensor.name == name && tensor.dtype == dtype)
        .ok_or_else(|| Error::Inference(format!("native StyleTTS2 omitted {name}")))
}

fn f32_values(tensor: &ffi::Tensor) -> Result<Vec<f32>> {
    if !tensor.data.len().is_multiple_of(4) {
        return Err(Error::Inference(format!("native tensor {} has invalid byte length", tensor.name)));
    }
    Ok(tensor.data.chunks_exact(4).map(|bytes| f32::from_ne_bytes(bytes.try_into().expect("four-byte chunk"))).collect())
}

fn i32_values(tensor: &ffi::Tensor) -> Result<Vec<i32>> {
    if !tensor.data.len().is_multiple_of(4) {
        return Err(Error::Inference(format!("native tensor {} has invalid byte length", tensor.name)));
    }
    Ok(tensor.data.chunks_exact(4).map(|bytes| i32::from_ne_bytes(bytes.try_into().expect("four-byte chunk"))).collect())
}

fn alignment(request: &ModelRequest, text: &PreparedText, durations: &[i32]) -> Option<Alignment> {
    if request.alignment_type == AlignmentType::None {
        return None;
    }
    let phonemes = phoneme_items(&text.phonemes, durations);
    let items = match request.alignment_type {
        AlignmentType::Phoneme => phonemes,
        AlignmentType::Char => character_items(&request.text, &text.mapping, &phonemes),
        AlignmentType::Word => word_items(&request.text, &text.mapping, &phonemes),
        AlignmentType::None => unreachable!("none alignment returned above"),
    };
    Some(Alignment { items, kind: request.alignment_type })
}

fn phoneme_items(phonemes: &str, durations: &[i32]) -> Vec<AlignmentItem> {
    let mut start_ms = 0;
    phonemes
        .chars()
        .enumerate()
        .zip(durations.iter().skip(1))
        .map(|((start, character), duration)| {
            let end_ms = start_ms + (*duration).max(0) as u64 * 25;
            let item = AlignmentItem { item: character.to_string(), char_start: start, char_end: start + 1, start_ms, end_ms };
            start_ms = end_ms;
            item
        })
        .collect()
}

fn word_items(text: &str, mapping: &[MappedItem], phonemes: &[AlignmentItem]) -> Vec<AlignmentItem> {
    let mut index = 0;
    let mut last_time = 0;
    mapping
        .iter()
        .filter_map(|mapped| {
            let segment = &phonemes[index..index + mapped.phoneme_count];
            index += mapped.phoneme_count;
            let (start_ms, end_ms) = segment.first().zip(segment.last()).map_or((last_time, last_time), |(first, last)| {
                last_time = last.end_ms;
                (first.start_ms, last.end_ms)
            });
            (mapped.original_start < mapped.original_end).then(|| AlignmentItem {
                item: text[mapped.original_start..mapped.original_end].to_owned(),
                char_start: mapped.original_start,
                char_end: mapped.original_end,
                start_ms,
                end_ms,
            })
        })
        .collect()
}

fn character_items(text: &str, mapping: &[MappedItem], phonemes: &[AlignmentItem]) -> Vec<AlignmentItem> {
    word_items(text, mapping, phonemes)
        .into_iter()
        .flat_map(|word| {
            let characters = word.item.char_indices().collect::<Vec<_>>();
            let count = characters.len().max(1) as u64;
            let duration = word.end_ms - word.start_ms;
            characters.into_iter().enumerate().map(move |(index, (offset, character))| AlignmentItem {
                item: character.to_string(),
                char_start: word.char_start + offset,
                char_end: word.char_start + offset + character.len_utf8(),
                start_ms: word.start_ms + duration * index as u64 / count,
                end_ms: word.start_ms + duration * (index as u64 + 1) / count,
            })
        })
        .collect()
}
