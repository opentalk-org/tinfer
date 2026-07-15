use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;
use std::time::Duration;

use crossbeam_channel::{Receiver, Sender, bounded};

use super::caller::{Call, callers};
use crate::models::Model;
use crate::{AudioChunk, Config, Error, ModelInfo, Result, StreamParams};

pub(crate) enum Delivery {
    Chunk(AudioChunk),
    End(bool),
    Error(Error),
}

pub(crate) enum Message {
    Load(crate::ModelConfig, Sender<Result<()>>),
    Unload(String, Sender<Result<()>>),
    Models(Sender<Result<Vec<ModelInfo>>>),
    Voices(String, Sender<Result<Vec<String>>>),
    Create(String, String, StreamParams, Option<String>, Sender<Result<(u64, Receiver<Delivery>)>>),
    Append(u64, String, Sender<Result<()>>),
    Wake(u64, Sender<Result<()>>),
    Force(u64, Sender<Result<()>>),
    Finish(u64, Sender<Result<()>>),
    Cancel(u64, Sender<Result<()>>),
    State(u64, Sender<Result<serde_json::Value>>),
    Close(u64, Sender<Result<()>>),
    Complete(Call, Result<Vec<crate::ModelOutput>>),
    Stop(Sender<Result<()>>),
}

#[derive(Clone)]
pub struct Engine {
    tx: Sender<Message>,
    join: Arc<Mutex<Option<JoinHandle<()>>>>,
    defaults: StreamParams,
}

pub struct Stream {
    id: u64,
    tx: Sender<Message>,
    rx: Receiver<Delivery>,
}

pub(super) type LoadedModel = (Arc<dyn Model>, usize, serde_json::Map<String, serde_json::Value>);

impl Engine {
    pub fn new(config: Config) -> Result<Self> {
        config.validate()?;
        let loaded = config
            .models
            .iter()
            .map(|model| {
                let settings = model.settings.as_object().expect("validated model settings are an object").clone();
                crate::models::load(model).map(|loaded| (loaded, model.max_batch, settings))
            })
            .collect::<Result<Vec<_>>>()?;
        let (tx, rx) = bounded(config.engine.queue_capacity);
        let (work_tx, work_rx) = bounded(config.engine.queue_capacity);
        let caller_count = config.models.len().max(1);
        let caller_joins = callers(caller_count, work_rx, tx.clone());
        let batch_wait = Duration::from_millis(config.engine.engine_timeout_ms);
        let defaults = config.defaults.stream_params();
        let join = std::thread::spawn(move || {
            super::coordinator::coordinate(loaded, rx, work_tx, batch_wait);
            for join in caller_joins {
                join.join().expect("caller thread must not panic");
            }
        });
        Ok(Self { tx, join: Arc::new(Mutex::new(Some(join))), defaults })
    }

    pub fn load_model(&self, config: crate::ModelConfig) -> Result<()> {
        request(&self.tx, |reply| Message::Load(config, reply))
    }

    pub fn unload_model(&self, model: &str) -> Result<()> {
        request(&self.tx, |reply| Message::Unload(model.into(), reply))
    }

    pub fn get_model_ids(&self) -> Result<Vec<String>> {
        Ok(request(&self.tx, Message::Models)?.into_iter().map(|info| info.model_id).collect())
    }

    pub fn get_model_infos(&self) -> Result<Vec<ModelInfo>> {
        request(&self.tx, Message::Models)
    }

    pub fn get_voice_ids(&self, model: &str) -> Result<Vec<String>> {
        request(&self.tx, |reply| Message::Voices(model.into(), reply))
    }

    pub fn stream_params(&self) -> StreamParams {
        self.defaults.clone()
    }

    pub fn create_stream(&self, model: &str, voice: &str, params: StreamParams) -> Result<Stream> {
        let (id, rx) = request(&self.tx, |reply| Message::Create(model.into(), voice.into(), params, None, reply))?;
        Ok(Stream { id, tx: self.tx.clone(), rx })
    }

    pub fn start_stream(&self, model: &str, voice: &str, text: &str, params: StreamParams) -> Result<Stream> {
        let (id, rx) = request(&self.tx, |reply| Message::Create(model.into(), voice.into(), params, Some(text.into()), reply))?;
        Ok(Stream { id, tx: self.tx.clone(), rx })
    }

    pub fn generate_full(&self, model: &str, voice: &str, text: &str, params: StreamParams) -> Result<AudioChunk> {
        let stream = self.create_stream(model, voice, params)?;
        stream.add_text(text)?;
        stream.force_generate()?;
        let chunks = stream.collect_audio()?;
        stream.close()?;
        AudioChunk::merge(chunks)
    }

    pub fn stop(&self) -> Result<()> {
        let Some(join) = self.join.lock().expect("engine join lock poisoned").take() else {
            return Ok(());
        };
        let result = request(&self.tx, Message::Stop);
        join.join().expect("coordinator thread must not panic");
        result
    }
}

impl Stream {
    pub fn add_text(&self, text: &str) -> Result<()> {
        request(&self.tx, |reply| Message::Append(self.id, text.into(), reply))
    }

    pub fn force_generate(&self) -> Result<()> {
        request(&self.tx, |reply| Message::Force(self.id, reply))
    }

    pub fn try_generate(&self) -> Result<()> {
        request(&self.tx, |reply| Message::Wake(self.id, reply))
    }

    pub fn finish(&self) -> Result<()> {
        request(&self.tx, |reply| Message::Finish(self.id, reply))
    }

    pub fn cancel(&self) -> Result<()> {
        request(&self.tx, |reply| Message::Cancel(self.id, reply))
    }

    pub fn recv(&self) -> Result<Option<AudioChunk>> {
        self.delivery(self.rx.recv().map_err(|_| Error::Shutdown)?)
    }

    pub(crate) fn recv_marked(&self) -> Result<(Option<AudioChunk>, bool)> {
        let delivery = self.rx.recv().map_err(|_| Error::Shutdown)?;
        let final_delivery = matches!(delivery, Delivery::End(true));
        Ok((self.delivery(delivery)?, final_delivery))
    }

    pub fn get_audio(&self) -> Result<Vec<AudioChunk>> {
        self.rx.try_iter().flat_map(|delivery| self.delivery(delivery).transpose()).collect()
    }

    pub fn get_state(&self) -> Result<serde_json::Value> {
        request(&self.tx, |reply| Message::State(self.id, reply))
    }

    fn delivery(&self, delivery: Delivery) -> Result<Option<AudioChunk>> {
        match delivery {
            Delivery::Chunk(chunk) => Ok(Some(chunk)),
            Delivery::End(_) => Ok(None),
            Delivery::Error(error) => Err(error),
        }
    }

    pub fn collect_audio(&self) -> Result<Vec<AudioChunk>> {
        std::iter::from_fn(|| self.recv().transpose()).collect()
    }

    pub fn close(&self) -> Result<()> {
        self.rx.try_iter().for_each(drop);
        request(&self.tx, |reply| Message::Close(self.id, reply))
    }
}

fn request<T>(engine: &Sender<Message>, message: impl FnOnce(Sender<Result<T>>) -> Message) -> Result<T> {
    let (tx, rx) = bounded(1);
    engine.send(message(tx)).map_err(|_| Error::Shutdown)?;
    rx.recv().map_err(|_| Error::Shutdown)?
}
