use std::fs;
use std::sync::Arc;

use serde::Deserialize;

use super::base::Model;
use super::base::native::{Handle, ffi, tensor};
use crate::{Alignment, AlignmentItem, AlignmentType, Error, ModelConfig, ModelInfo, ModelOperation, ModelOutput, ModelRequest, Result};

#[derive(Deserialize)]
struct Id {
    id: String,
}

#[derive(Deserialize)]
struct Manifest {
    sample_rate: u32,
    language: String,
    languages: Vec<Id>,
    voices: Vec<Id>,
}

struct Stub {
    info: ModelInfo,
    voices: Vec<String>,
    sample_rate: u32,
    native: Handle,
}

pub fn load(config: &ModelConfig) -> Result<Arc<dyn Model>> {
    let source = fs::read_to_string(config.path.join("manifest.toml")).map_err(|error| Error::Validation(error.to_string()))?;
    let manifest: Manifest = toml::from_str(&source).map_err(|error| Error::Validation(format!("invalid stub manifest: {error}")))?;
    if manifest.sample_rate == 0
        || !manifest.languages.iter().any(|language| language.id == manifest.language)
        || manifest.voices.is_empty()
    {
        return Err(Error::Validation("invalid stub manifest".into()));
    }
    let model_id = config.id.clone();
    let native = Handle::stub()?;
    Ok(Arc::new(Stub {
        info: ModelInfo {
            model_id: model_id.clone(),
            supported_languages: manifest.languages.into_iter().map(|language| language.id).collect(),
            default_language: manifest.language,
        },
        voices: manifest.voices.into_iter().map(|voice| voice.id).collect(),
        sample_rate: manifest.sample_rate,
        native,
    }))
}

impl Model for Stub {
    fn info(&self) -> &ModelInfo {
        &self.info
    }

    fn voices(&self) -> &[String] {
        &self.voices
    }

    fn generate_batch(&self, batch: &[ModelRequest]) -> Result<Vec<ModelOutput>> {
        let mut values = Vec::new();
        let mut offsets = vec![0];
        let mut ranges = Vec::with_capacity(batch.len());
        for request in batch {
            let start = match request.operation {
                ModelOperation::Start => 0,
                ModelOperation::Continue => request.state["stub_character_offset"]
                    .as_u64()
                    .ok_or_else(|| Error::Validation("stub continuation is missing its character offset".into()))?
                    as usize,
            };
            let limit = request.params["characters_per_call"].as_u64().map_or(usize::MAX, |value| value as usize);
            let characters = request.text.chars().skip(start).take(limit).collect::<Vec<_>>();
            let end = start + characters.len();
            values.extend(characters.into_iter().map(|character| i64::from(u32::from(character))));
            offsets.push(u32::try_from(values.len()).map_err(|_| Error::Validation("native batch contains too many characters".into()))?);
            ranges.push((start, end, end == request.text.chars().count()));
        }
        let value_bytes = values.iter().flat_map(|value| value.to_ne_bytes()).collect();
        let offset_bytes = offsets.iter().flat_map(|offset| offset.to_ne_bytes()).collect();
        let mut native = self.native.generate(vec![
            tensor("tokens", ffi::DType::I64, vec![values.len() as i64], value_bytes),
            tensor("offsets", ffi::DType::I32, vec![offsets.len() as i64], offset_bytes),
        ])?;
        let offsets =
            native.pop().filter(|value| value.name == "offsets").ok_or_else(|| Error::Inference("native stub omitted offsets".into()))?;
        let audio =
            native.pop().filter(|value| value.name == "audio").ok_or_else(|| Error::Inference("native stub omitted audio".into()))?;
        let offsets =
            offsets.data.chunks_exact(4).map(|bytes| u32::from_ne_bytes(bytes.try_into().expect("four-byte chunk"))).collect::<Vec<_>>();
        let audio =
            audio.data.chunks_exact(4).map(|bytes| f32::from_ne_bytes(bytes.try_into().expect("four-byte chunk"))).collect::<Vec<_>>();
        batch
            .iter()
            .enumerate()
            .zip(ranges)
            .map(|((batch_index, request), (character_start, character_end, complete))| {
                if !self.voices.contains(&request.voice_id) {
                    return Err(Error::Validation(format!("voice not found: {}", request.voice_id)));
                }
                let start = offsets[batch_index] as usize;
                let end = offsets[batch_index + 1] as usize;
                let alignment = (request.alignment_type != AlignmentType::None).then(|| Alignment {
                    kind: request.alignment_type,
                    items: request
                        .text
                        .char_indices()
                        .enumerate()
                        .filter(|(index, _)| *index >= character_start && *index < character_end)
                        .map(|(_, (start, character))| AlignmentItem {
                            item: character.to_string(),
                            char_start: start,
                            char_end: start + character.len_utf8(),
                            start_ms: 0,
                            end_ms: 1,
                        })
                        .collect(),
                });
                Ok(ModelOutput {
                    audio: audio[start..end].to_vec(),
                    sample_rate: self.sample_rate,
                    alignment,
                    state: if complete {
                        serde_json::json!({"text": request.text})
                    } else {
                        serde_json::json!({"text": request.text, "stub_character_offset": character_end})
                    },
                    complete,
                })
            })
            .collect()
    }

    fn close_stream(&self, _stream_id: u64) -> Result<()> {
        Ok(())
    }
}
