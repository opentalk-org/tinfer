use std::collections::{BTreeMap, HashMap};
use std::sync::{Arc, Mutex};

use super::base::Model;
use super::base::native::Handle;
use crate::{Backend, Device, Error, ModelConfig, ModelInfo, ModelOutput, ModelRequest, Result};
use espeak_align_core::Engine;

mod manifest;
mod postprocessing;
mod preprocessing;
mod text;

#[cfg(test)]
mod tests;

use manifest::Manifest;
use preprocessing::{parameters, prepare};

struct StyleTts2 {
    info: ModelInfo,
    voices: Vec<String>,
    manifest: Manifest,
    voice_cache: Mutex<HashMap<String, Vec<f32>>>,
    phonemizers: Mutex<HashMap<String, Engine>>,
    native: Handle,
}

pub fn load(config: &ModelConfig) -> Result<Arc<dyn Model>> {
    if config.backend == Backend::Tensorrt && config.device == Device::Cpu {
        return Err(Error::Validation("StyleTTS2 TensorRT requires a CUDA device".into()));
    }
    let manifest = Manifest::load(&config.path)?;
    let device = match config.device {
        Device::Cpu => -1,
        Device::Cuda(index) => i32::try_from(index).map_err(|_| Error::Validation("CUDA device index exceeds native range".into()))?,
        Device::Auto if config.backend == Backend::Tensorrt || cfg!(feature = "onnx-cuda") => 0,
        Device::Auto => -1,
    };
    let backend = match config.backend {
        Backend::Onnx => 0,
        Backend::Tensorrt => 1,
    };
    let native = Handle::styletts2(
        config.path.to_str().ok_or_else(|| Error::Validation("StyleTTS2 path must be UTF-8".into()))?,
        &manifest.architecture_id,
        backend,
        device,
    )?;
    let voices = manifest.voices.keys().cloned().collect();
    Ok(Arc::new(StyleTts2 {
        info: ModelInfo {
            model_id: config.id.clone(),
            supported_languages: manifest.supported_languages.clone(),
            default_language: manifest.default_language.clone(),
        },
        voices,
        manifest,
        voice_cache: Mutex::new(HashMap::new()),
        phonemizers: Mutex::new(HashMap::new()),
        native,
    }))
}

impl Model for StyleTts2 {
    fn info(&self) -> &ModelInfo {
        &self.info
    }

    fn voices(&self) -> &[String] {
        &self.voices
    }

    fn generate_batch(&self, batch: &[ModelRequest]) -> Result<Vec<ModelOutput>> {
        let mut groups = BTreeMap::<(usize, u32), Vec<usize>>::new();
        for (index, request) in batch.iter().enumerate() {
            let params = parameters(request)?;
            groups.entry((params.diffusion_steps, params.embedding_scale.to_bits())).or_default().push(index);
        }
        let mut output = vec![None; batch.len()];
        for indexes in groups.into_values() {
            let requests = indexes.iter().map(|index| batch[*index].clone()).collect::<Vec<_>>();
            let prepared = prepare(&requests, &self.manifest, &self.voice_cache, &self.phonemizers)?;
            let tensors = self.native.generate(prepared.tensors)?;
            let generated = postprocessing::finish(&requests, prepared.texts, tensors, self.manifest.sample_rate)?;
            for (index, item) in indexes.into_iter().zip(generated) {
                output[index] = Some(item);
            }
        }
        output.into_iter().map(|item| item.ok_or_else(|| Error::Inference("StyleTTS2 omitted a batch item".into()))).collect()
    }
}
