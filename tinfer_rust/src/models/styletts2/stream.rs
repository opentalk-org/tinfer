use std::collections::HashMap;

use crate::models::base::native::ffi;
use crate::{Alignment, AlignmentItem, ModelOutput, ModelRequest, Result};

use super::postprocessing::{alignment, f32_values, find, i32_values};
use super::text::PreparedText;

pub(super) struct State {
    request: ModelRequest,
    text: PreparedText,
    durations: Vec<i32>,
    style: serde_json::Value,
}

pub(super) type Streams = HashMap<u64, State>;

pub(super) fn started(
    requests: &[ModelRequest],
    texts: Vec<PreparedText>,
    tensors: Vec<ffi::Tensor>,
    streams: &mut Streams,
    sample_rate: u32,
) -> Result<Vec<ModelOutput>> {
    let duration_tensor = find(&tensors, "durations", ffi::DType::I32)?;
    let style_tensor = find(&tensors, "style", ffi::DType::F32)?;
    let durations = i32_values(duration_tensor)?;
    let styles = f32_values(style_tensor)?;
    let batch = requests.len();
    if texts.len() != batch
        || duration_tensor.shape.len() != 2
        || duration_tensor.shape[0] != batch as i64
        || durations.len() % batch != 0
        || style_tensor.shape.as_slice() != [batch as i64, 256]
        || styles.len() != batch * 256
    {
        return Err(crate::Error::Inference("native StyleTTS2 start output differs from batch".into()));
    }
    if requests.iter().any(|request| streams.contains_key(&request.stream_id)) {
        return Err(crate::Error::Inference("StyleTTS2 stream already has active generation".into()));
    }
    let stride = durations.len() / requests.len();
    requests
        .iter()
        .zip(texts)
        .enumerate()
        .map(|(index, (request, text))| {
            let style_start = index * 256;
            let style = serde_json::json!({
                "previous_style_vector": &styles[style_start..style_start + 256],
                "text": request.text,
            });
            let state =
                State { request: request.clone(), text, durations: durations[index * stride..(index + 1) * stride].to_vec(), style };
            streams.insert(request.stream_id, state);
            Ok(ModelOutput { audio: Vec::new(), sample_rate, alignment: None, state: request.state.clone(), complete: false })
        })
        .collect()
}

pub(super) fn continued(
    requests: &[ModelRequest],
    tensors: Vec<ffi::Tensor>,
    streams: &mut Streams,
    sample_rate: u32,
) -> Result<Vec<ModelOutput>> {
    let audio = f32_values(find(&tensors, "audio", ffi::DType::F32)?)?;
    let offsets = i32_values(find(&tensors, "audio_offsets", ffi::DType::I32)?)?;
    let starts = i32_values(find(&tensors, "frame_starts", ffi::DType::I32)?)?;
    let counts = i32_values(find(&tensors, "frame_counts", ffi::DType::I32)?)?;
    let complete = i32_values(find(&tensors, "complete", ffi::DType::I32)?)?;
    let batch = requests.len();
    if offsets.len() != batch + 1
        || starts.len() != batch
        || counts.len() != batch
        || complete.len() != batch
        || offsets.first() != Some(&0)
        || offsets.last().copied() != Some(audio.len() as i32)
        || offsets.windows(2).any(|pair| pair[0] < 0 || pair[0] > pair[1])
        || starts.iter().chain(&counts).any(|value| *value < 0)
    {
        return Err(crate::Error::Inference("native StyleTTS2 continuation output differs from batch".into()));
    }
    let mut output = Vec::with_capacity(requests.len());
    for (index, request) in requests.iter().enumerate() {
        let state = streams
            .get(&request.stream_id)
            .ok_or_else(|| crate::Error::Inference("StyleTTS2 continuation has no active generation".into()))?;
        let item_alignment = alignment(&state.request, &state.text, &state.durations)
            .map(|value| slice_alignment(value, starts[index] as u64 * 25, (starts[index] + counts[index]) as u64 * 25));
        output.push(ModelOutput {
            audio: audio[offsets[index] as usize..offsets[index + 1] as usize].to_vec(),
            sample_rate,
            alignment: item_alignment,
            state: state.style.clone(),
            complete: complete[index] != 0,
        });
        if complete[index] != 0 {
            streams.remove(&request.stream_id);
        }
    }
    Ok(output)
}

fn slice_alignment(alignment: Alignment, start_ms: u64, end_ms: u64) -> Alignment {
    let items = alignment
        .items
        .into_iter()
        .filter(|item| item.end_ms > start_ms && item.start_ms < end_ms)
        .map(|item| AlignmentItem {
            start_ms: item.start_ms.saturating_sub(start_ms),
            end_ms: item.end_ms.min(end_ms).saturating_sub(start_ms),
            ..item
        })
        .collect();
    Alignment { items, kind: alignment.kind }
}
