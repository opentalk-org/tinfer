use std::time::Instant;

use crossbeam_channel::Sender;

use super::caller::{Call, Dispatch};
use super::engine::Delivery;
use super::registry::Registry;
use crate::{Error, ModelOutput, ModelRequest, Result, StreamParams};

pub(crate) struct Request {
    pub id: u64,
    pub model: String,
    pub voice: String,
    pub language: String,
    pub params: StreamParams,
    pub text: String,
    pub committed: usize,
    pub chunk_index: u64,
    pub pending: usize,
    pub forced: bool,
    pub finished: bool,
    pub cancelled: bool,
    pub deadline: Option<Instant>,
    pub state: serde_json::Value,
    pub output: Sender<Delivery>,
}

impl Request {
    pub fn new(id: u64, model: String, voice: String, language: String, params: StreamParams, output: Sender<Delivery>) -> Self {
        Self {
            id,
            model,
            voice,
            language,
            params,
            text: String::new(),
            committed: 0,
            chunk_index: 0,
            pending: 0,
            forced: false,
            finished: false,
            cancelled: false,
            deadline: None,
            state: serde_json::Value::Object(Default::default()),
            output,
        }
    }

    pub fn append(&mut self, text: String) {
        if !text.trim().is_empty() {
            if self.text[self.committed..].trim().is_empty() {
                self.deadline = Some(Instant::now() + self.params.timeout);
            }
            self.text.push_str(&text);
        }
    }

    pub fn finish(&mut self) {
        self.forced = true;
        self.finished = true;
        if self.pending == 0 && self.committed == self.text.len() {
            let _ = self.output.send(Delivery::End(true));
        }
    }

    fn ready(&self, now: Instant) -> bool {
        let trigger = self
            .params
            .chunk_length_schedule
            .get(self.chunk_index as usize)
            .or_else(|| self.params.chunk_length_schedule.last())
            .copied()
            .unwrap_or(1);
        !self.cancelled
            && self.pending == 0
            && !self.text[self.committed..].trim().is_empty()
            && (self.forced
                || self.text[self.committed..].chars().count() > trigger
                || self.deadline.is_some_and(|deadline| deadline <= now))
    }
}

pub(crate) fn find_request(requests: &mut [Request], id: u64) -> Result<&mut Request> {
    requests.iter_mut().find(|request| request.id == id).ok_or_else(|| Error::Validation("stream is closed".into()))
}

pub(crate) fn next_wait(requests: &[Request]) -> std::time::Duration {
    let now = Instant::now();
    requests
        .iter()
        .filter_map(|request| request.deadline)
        .min()
        .map_or(std::time::Duration::from_secs(3_600), |deadline| deadline.saturating_duration_since(now))
}

pub(crate) fn unload(requests: &mut Vec<Request>, model: &str) {
    requests.retain(|request| {
        if request.model != model {
            return true;
        }
        let _ = request.output.send(Delivery::Error(Error::Catalog(format!("model unloaded: {model}"))));
        false
    });
}

pub(crate) fn dispatch(registry: &mut Registry, requests: &mut [Request], work: &Sender<Call>) {
    let now = Instant::now();
    let mut models = requests.iter().filter(|request| request.ready(now)).map(|request| request.model.clone()).collect::<Vec<_>>();
    models.sort();
    models.dedup();
    for model_id in models {
        let Some((entry, model, max_batch)) = registry.choose(&model_id) else {
            continue;
        };
        let mut batch = Vec::new();
        let mut dispatches = Vec::new();
        for request in requests.iter_mut().filter(|request| request.model == model_id && request.ready(now)).take(max_batch) {
            let start = request.committed;
            let end = start + next_chunk(request).len();
            let text = request.text[start..end].to_owned();
            request.committed = end;
            request.pending += 1;
            request.forced = request.committed < request.text.len();
            request.deadline = None;
            batch.push(ModelRequest {
                text,
                voice_id: request.voice.clone(),
                params: request.params.model.clone(),
                state: request.state.clone(),
                alignment_type: request.params.alignment_type,
            });
            dispatches.push(Dispatch {
                request_id: request.id,
                chunk_index: request.chunk_index,
                text_start: start,
                text_end: request.committed,
            });
            request.chunk_index += 1;
        }
        work.send(Call { entry, model, batch, dispatches }).expect("caller pool exists while coordinator runs");
    }
}

fn next_chunk(request: &Request) -> &str {
    let pending = &request.text[request.committed..];
    let index = request.chunk_index as usize;
    let trigger = request
        .params
        .chunk_length_schedule
        .get(index)
        .or_else(|| request.params.chunk_length_schedule.last())
        .copied()
        .expect("chunk schedule is validated");
    let no_split = request.params.chunk_length_schedule.get(index + 1).copied().unwrap_or(trigger + trigger / 3);
    if pending.chars().count() <= no_split {
        return pending;
    }
    let mut end = 0;
    for sentence in sentencex::segment(&request.language, pending) {
        if end > 0 && pending[..end].chars().count() + sentence.chars().count() > trigger {
            break;
        }
        end += sentence.len();
        if pending[..end].chars().count() >= trigger {
            break;
        }
    }
    if end == 0 || end == pending.len() {
        end = pending.char_indices().nth(trigger).map_or(pending.len(), |(byte, _)| byte);
        if let Some(space) = pending[..end].rfind(char::is_whitespace) {
            end = space + pending[space..].chars().next().expect("space exists").len_utf8();
        }
    }
    &pending[..end]
}

pub(crate) fn complete(registry: &mut Registry, requests: &mut [Request], call: Call, result: Result<Vec<ModelOutput>>) {
    registry.release(call.entry);
    let outputs = match result {
        Ok(outputs) if outputs.len() == call.dispatches.len() => outputs,
        Ok(_) => {
            fail(requests, &call, Error::Inference("model output count differs from batch".into()));
            return;
        }
        Err(error) => {
            fail(requests, &call, error);
            return;
        }
    };
    for (dispatch, output) in call.dispatches.into_iter().zip(outputs) {
        let Some(request) = requests.iter_mut().find(|request| request.id == dispatch.request_id) else {
            continue;
        };
        request.pending -= 1;
        request.state = output.state;
        let _ = request.output.send(Delivery::Chunk(crate::AudioChunk {
            audio: output.audio,
            sample_rate: output.sample_rate,
            chunk_index: dispatch.chunk_index,
            text_span: dispatch.text_start..dispatch.text_end,
            alignment: output.alignment,
        }));
        if request.pending == 0 && request.committed == request.text.len() {
            let _ = request.output.send(Delivery::End(request.finished));
        }
    }
}

fn fail(requests: &mut [Request], call: &Call, error: Error) {
    for dispatch in &call.dispatches {
        if let Some(request) = requests.iter_mut().find(|request| request.id == dispatch.request_id) {
            request.pending -= 1;
            let _ = request.output.send(Delivery::Error(error.clone()));
        }
    }
}
