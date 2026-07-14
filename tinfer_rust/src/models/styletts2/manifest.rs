use std::collections::{BTreeMap, HashSet};
use std::fs::{self, File};
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};

use serde::Deserialize;

use crate::{Error, Result};

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Manifest {
    pub architecture_id: String,
    pub sample_rate: u32,
    pub default_language: String,
    pub supported_languages: Vec<String>,
    pub symbols: Vec<String>,
    #[serde(skip)]
    pub voices: BTreeMap<String, PathBuf>,
}

impl Manifest {
    pub fn load(root: &Path) -> Result<Self> {
        let source = fs::read_to_string(root.join("model.toml")).map_err(|error| Error::Validation(error.to_string()))?;
        let mut manifest: Self =
            toml::from_str(&source).map_err(|error| Error::Validation(format!("invalid StyleTTS2 manifest: {error}")))?;
        let unique_languages = manifest.supported_languages.iter().collect::<HashSet<_>>();
        if manifest.architecture_id.is_empty()
            || manifest.sample_rate != 24_000
            || manifest.supported_languages.is_empty()
            || unique_languages.len() != manifest.supported_languages.len()
            || !manifest.supported_languages.contains(&manifest.default_language)
            || manifest.symbols.first().map(String::as_str) != Some("$")
            || manifest.symbols.iter().any(String::is_empty)
            || manifest.symbols.iter().any(|symbol| symbol.chars().count() != 1)
        {
            return Err(Error::Validation("invalid StyleTTS2 manifest invariants".into()));
        }
        manifest.voices = fs::read_dir(root.join("voices"))
            .map_err(|error| Error::Validation(error.to_string()))?
            .map(|entry| entry.map_err(|error| Error::Validation(error.to_string())))
            .collect::<Result<Vec<_>>>()?
            .into_iter()
            .filter(|entry| entry.path().extension().and_then(|extension| extension.to_str()) == Some("tinf"))
            .map(|entry| {
                let path = entry.path();
                let id = path.file_stem().expect("voice path has a stem").to_string_lossy().into_owned();
                (id, path)
            })
            .collect();
        if manifest.voices.is_empty() {
            return Err(Error::Validation("StyleTTS2 export contains no voices".into()));
        }
        Ok(manifest)
    }

    pub fn load_voice(&self, voice: &str) -> Result<Vec<f32>> {
        let path = self.voices.get(voice).ok_or_else(|| Error::Catalog(format!("voice not found: {voice}")))?;
        let mut input = File::open(path).map_err(|error| Error::Validation(error.to_string()))?;
        let mut magic = [0; 4];
        input.read_exact(&mut magic).map_err(|error| Error::Validation(error.to_string()))?;
        if &magic != b"TINF" || read_i32(&mut input)? != 1 {
            return Err(Error::Validation(format!("invalid StyleTTS2 voice: {voice}")));
        }
        let name_length = read_i32(&mut input)?;
        if name_length < 1 {
            return Err(Error::Validation(format!("invalid StyleTTS2 voice: {voice}")));
        }
        input.seek(SeekFrom::Current(i64::from(name_length))).map_err(|error| Error::Validation(error.to_string()))?;
        let dtype = read_i32(&mut input)?;
        let rank = read_i32(&mut input)?;
        let dimensions = (0..rank).map(|_| read_i64(&mut input)).collect::<Result<Vec<_>>>()?;
        if dimensions.iter().product::<i64>() != 256 || !matches!(dtype, 0 | 1) {
            return Err(Error::Validation(format!("StyleTTS2 voice {voice} must contain 256 floats")));
        }
        let mut bytes = vec![0; 256 * if dtype == 0 { 2 } else { 4 }];
        input.read_exact(&mut bytes).map_err(|error| Error::Validation(error.to_string()))?;
        Ok(if dtype == 0 {
            bytes.chunks_exact(2).map(|value| half::f16::from_le_bytes(value.try_into().expect("two-byte chunk")).to_f32()).collect()
        } else {
            bytes.chunks_exact(4).map(|value| f32::from_le_bytes(value.try_into().expect("four-byte chunk"))).collect()
        })
    }
}

fn read_i32(input: &mut File) -> Result<i32> {
    let mut bytes = [0; 4];
    input.read_exact(&mut bytes).map_err(|error| Error::Validation(error.to_string()))?;
    Ok(i32::from_le_bytes(bytes))
}

fn read_i64(input: &mut File) -> Result<i64> {
    let mut bytes = [0; 8];
    input.read_exact(&mut bytes).map_err(|error| Error::Validation(error.to_string()))?;
    Ok(i64::from_le_bytes(bytes))
}
