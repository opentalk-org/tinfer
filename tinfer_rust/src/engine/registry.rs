use std::collections::HashSet;
use std::sync::Arc;

use crate::models::Model;
use crate::{Error, ModelInfo, Result};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct EntryId(u64);

pub(crate) struct Loaded {
    pub id: EntryId,
    pub model: Arc<dyn Model>,
    pub busy: bool,
    pub max_batch: usize,
}

pub(crate) struct Registry {
    pub entries: Vec<Loaded>,
    next_id: u64,
}

impl Default for Registry {
    fn default() -> Self {
        Self { entries: Vec::new(), next_id: 1 }
    }
}

impl Registry {
    pub fn add(&mut self, model: Arc<dyn Model>, max_batch: usize) {
        let id = EntryId(self.next_id);
        self.next_id += 1;
        self.entries.push(Loaded { id, model, busy: false, max_batch });
    }

    pub fn unload(&mut self, model: &str) -> Result<()> {
        let before = self.entries.len();
        self.entries.retain(|entry| entry.model.info().model_id != model);
        if self.entries.len() == before {
            return Err(Error::Catalog(format!("model not found: {model}")));
        }
        Ok(())
    }

    pub fn validate(&self, model: &str, voice: &str) -> Result<String> {
        let entry = self
            .entries
            .iter()
            .find(|entry| entry.model.info().model_id == model)
            .ok_or_else(|| Error::Catalog(format!("model not found: {model}")))?;
        if entry.model.voices().iter().all(|item| item != voice) {
            return Err(Error::Catalog(format!("voice not found: {voice}")));
        }
        Ok(entry.model.info().default_language.clone())
    }

    pub fn choose(&mut self, model: &str) -> Option<(EntryId, Arc<dyn Model>, usize)> {
        let entry = self.entries.iter_mut().find(|entry| entry.model.info().model_id == model && !entry.busy)?;
        entry.busy = true;
        Some((entry.id, entry.model.clone(), entry.max_batch))
    }

    pub fn release(&mut self, id: EntryId) {
        if let Some(entry) = self.entries.iter_mut().find(|entry| entry.id == id) {
            entry.busy = false;
        }
    }

    pub fn infos(&self) -> Vec<ModelInfo> {
        let mut seen = HashSet::new();
        self.entries
            .iter()
            .filter_map(|entry| {
                let info = entry.model.info();
                seen.insert(info.model_id.clone()).then(|| info.clone())
            })
            .collect()
    }

    pub fn voices(&self, model: &str) -> Result<Vec<String>> {
        self.entries
            .iter()
            .find(|entry| entry.model.info().model_id == model)
            .map(|entry| entry.model.voices().to_vec())
            .ok_or_else(|| Error::Catalog(format!("model not found: {model}")))
    }
}
