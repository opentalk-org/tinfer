use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;

use crossbeam_channel::{bounded, Receiver, RecvTimeoutError, Sender};

use super::caller::{callers, Call};
use super::registry::Registry;
use super::scheduler::{dispatch, find_request, next_wait, unload, Request};
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
    Create(String, String, StreamParams, Sender<Result<(u64, Receiver<Delivery>)>>),
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
}

pub struct Stream {
    id: u64,
    tx: Sender<Message>,
    rx: Receiver<Delivery>,
}

impl Engine {
    pub fn new(config: Config) -> Result<Self> {
        config.validate()?;
        let loaded = config
            .models
            .iter()
            .map(|model| crate::models::load(model).map(|loaded| (loaded, model.max_batch)))
            .collect::<Result<Vec<_>>>()?;
        let (tx, rx) = bounded(config.queue_capacity);
        let (work_tx, work_rx) = bounded(config.queue_capacity);
        let caller_count = config.models.len().max(1);
        let caller_joins = callers(caller_count, work_rx, tx.clone());
        let join = std::thread::spawn(move || {
            coordinate(loaded, rx, work_tx);
            for join in caller_joins {
                join.join().expect("caller thread must not panic");
            }
        });
        Ok(Self { tx, join: Arc::new(Mutex::new(Some(join))) })
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

    pub fn create_stream(&self, model: &str, voice: &str, params: StreamParams) -> Result<Stream> {
        let (id, rx) = request(&self.tx, |reply| Message::Create(model.into(), voice.into(), params, reply))?;
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

fn coordinate(loaded: Vec<(Arc<dyn Model>, usize)>, rx: Receiver<Message>, work: Sender<Call>) {
    let mut registry = Registry::default();
    for (model, max_batch) in loaded {
        registry.add(model, max_batch);
    }
    let mut requests = Vec::<Request>::new();
    let mut next_id = 1;
    loop {
        let stop = match rx.recv_timeout(next_wait(&requests)) {
            Ok(message) => process(message, &mut registry, &mut requests, &mut next_id),
            Err(RecvTimeoutError::Timeout) => false,
            Err(RecvTimeoutError::Disconnected) => return,
        };
        while let Ok(message) = rx.try_recv() {
            if process(message, &mut registry, &mut requests, &mut next_id) {
                return;
            }
        }
        dispatch(&mut registry, &mut requests, &work);
        if stop {
            return;
        }
    }
}

fn process(message: Message, registry: &mut Registry, requests: &mut Vec<Request>, next_id: &mut u64) -> bool {
    match message {
        Message::Load(config, reply) => {
            let result = crate::models::load(&config).map(|model| {
                registry.add(model, config.max_batch);
            });
            let _ = reply.send(result);
        }
        Message::Unload(model, reply) => {
            let result = registry.unload(&model);
            if result.is_ok() {
                unload(requests, &model);
            }
            let _ = reply.send(result);
        }
        Message::Models(reply) => {
            let _ = reply.send(Ok(registry.infos()));
        }
        Message::Voices(model, reply) => {
            let _ = reply.send(registry.voices(&model));
        }
        Message::Create(model, voice, mut params, reply) => {
            let result = registry.validate(&model, &voice).and_then(|default_language| {
                let language =
                    params.model["language"].as_str().filter(|language| !language.is_empty()).unwrap_or(&default_language).to_owned();
                params.model["language"] = serde_json::Value::String(language.clone());
                let id = *next_id;
                *next_id += 1;
                let (tx, rx) = bounded(2);
                requests.push(Request::new(id, model, voice, language, params, tx)?);
                Ok((id, rx))
            });
            let _ = reply.send(result);
        }
        Message::Append(id, text, reply) => {
            let result = find_request(requests, id).map(|request| request.append(text));
            let _ = reply.send(result);
        }
        Message::Wake(id, reply) => {
            let _ = reply.send(find_request(requests, id).map(drop));
        }
        Message::Force(id, reply) => {
            let result = find_request(requests, id).map(|request| request.forced = true);
            let _ = reply.send(result);
        }
        Message::Finish(id, reply) => {
            let result = find_request(requests, id).map(Request::finish);
            let _ = reply.send(result);
        }
        Message::Cancel(id, reply) => {
            let result = find_request(requests, id).map(|request| {
                request.cancelled = true;
                request.text.clear();
                request.committed = 0;
                request.committed_chars = 0;
                request.prepared.clear();
                request.deadline = None;
                request.forced = false;
                request.pending = 0;
                request.nonce = request.nonce.wrapping_add(1);
                let _ = request.output.send(Delivery::Error(Error::Cancelled));
            });
            let _ = reply.send(result);
        }
        Message::State(id, reply) => {
            let _ = reply.send(find_request(requests, id).map(|request| request.state.clone()));
        }
        Message::Close(id, reply) => {
            requests.retain(|request| request.id != id);
            let _ = reply.send(Ok(()));
        }
        Message::Complete(call, result) => super::scheduler::complete(registry, requests, call, result),
        Message::Stop(reply) => {
            for request in requests.drain(..) {
                let _ = request.output.send(Delivery::Error(Error::Shutdown));
            }
            let _ = reply.send(Ok(()));
            return true;
        }
    }
    false
}
