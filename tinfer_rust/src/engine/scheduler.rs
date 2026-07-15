use std::collections::{HashMap, VecDeque};
use std::time::{Duration, Instant};

use crossbeam_channel::Sender;

use super::caller::{Call, Dispatch};
use super::chunker::{Chunker, PreparedChunk};
use super::engine::Delivery;
use super::registry::{EntryId, Registry};
use crate::{Error, ModelOperation, ModelOutput, ModelRequest, Result, StreamParams};

pub(crate) struct ActiveChunk {
    pub entry: EntryId,
    pub text: String,
    pub span: std::ops::Range<usize>,
}

pub(crate) struct Request {
    pub id: u64,
    pub model: String,
    pub voice: String,
    pub params: StreamParams,
    pub text: String,
    pub committed: usize,
    pub committed_chars: usize,
    pub text_index: usize,
    pub output_index: u64,
    pub pending: usize,
    pub forced: bool,
    pub finished: bool,
    pub cancelled: bool,
    pub deadline: Option<Instant>,
    pub batch_at: Option<Instant>,
    pub prepared: VecDeque<PreparedChunk>,
    pub active: Option<ActiveChunk>,
    pub created_at: Instant,
    pub start_time: Option<Instant>,
    pub collected_time: Duration,
    pub nonce: u64,
    pub chunker: Chunker,
    pub state: serde_json::Value,
    pub output: Sender<Delivery>,
}

impl Request {
    pub fn new(id: u64, model: String, voice: String, language: String, params: StreamParams, output: Sender<Delivery>) -> Result<Self> {
        Ok(Self {
            id,
            model,
            voice,
            params,
            text: String::new(),
            committed: 0,
            committed_chars: 0,
            text_index: 0,
            output_index: 0,
            pending: 0,
            forced: false,
            finished: false,
            cancelled: false,
            deadline: None,
            batch_at: None,
            prepared: VecDeque::new(),
            active: None,
            created_at: Instant::now(),
            start_time: None,
            collected_time: Duration::ZERO,
            nonce: 0,
            chunker: Chunker::new(&language)?,
            state: serde_json::Value::Object(Default::default()),
            output,
        })
    }

    pub fn append(&mut self, text: String) {
        if text.trim().is_empty() {
            return;
        }
        if self.text[self.committed..].trim().is_empty() {
            self.deadline = Some(Instant::now() + self.params.timeout);
        }
        self.text.push_str(&text);
    }

    pub fn finish(&mut self) {
        self.forced = true;
        self.finished = true;
        if self.pending == 0 && self.active.is_none() && self.committed == self.text.len() {
            let _ = self.output.send(Delivery::End(true));
        }
    }

    fn prepare(&mut self, now: Instant, batch_wait: Duration) -> Result<bool> {
        if self.cancelled || self.pending > 0 {
            return Ok(false);
        }
        if self.active.is_some() {
            return Ok(true);
        }
        if !self.prepared.is_empty() {
            return Ok(true);
        }
        let pending = &self.text[self.committed..];
        if pending.trim().is_empty() {
            return Ok(false);
        }
        let index = self.text_index;
        let trigger = self.params.chunk_length_schedule[index.min(self.params.chunk_length_schedule.len() - 1)];
        let trigger_now = if self.forced {
            self.forced = false;
            true
        } else {
            pending.chars().count() > trigger || self.deadline.is_some_and(|deadline| deadline <= now)
        };
        if !trigger_now {
            return Ok(false);
        }
        let base = self.committed;
        self.prepared = self
            .chunker
            .prepare(pending, self.committed_chars, index, &self.params.chunk_length_schedule)?
            .into_iter()
            .map(|mut chunk| {
                chunk.bytes += base;
                chunk
            })
            .collect();
        self.batch_at = Some(now + batch_wait);
        Ok(true)
    }

    fn priority(&self, now: Instant) -> u128 {
        let first_audio = (self.output_index == 0) as u128 * 2_000_000_000_000_000_000;
        let realtime =
            self.start_time.is_some_and(|start| now.duration_since(start) > self.collected_time) as u128 * 1_000_000_000_000_000_000;
        first_audio + realtime + now.duration_since(self.created_at).as_nanos()
    }
}

pub(crate) fn find_request(requests: &mut [Request], id: u64) -> Result<&mut Request> {
    requests.iter_mut().find(|request| request.id == id).ok_or_else(|| Error::Validation("stream is closed".into()))
}

pub(crate) fn next_wait(requests: &[Request]) -> Duration {
    let now = Instant::now();
    requests
        .iter()
        .filter(|request| request.pending == 0 && request.active.is_none())
        .filter_map(|request| request.deadline)
        .chain(requests.iter().filter_map(|request| request.batch_at))
        .min()
        .map_or(Duration::from_secs(3_600), |deadline| deadline.saturating_duration_since(now))
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

pub(crate) fn dispatch(registry: &mut Registry, requests: &mut [Request], work: &Sender<Call>, batch_wait: Duration) {
    let now = Instant::now();
    let mut ready: HashMap<String, Vec<usize>> = HashMap::new();
    let mut model_order = Vec::new();
    for (index, request) in requests.iter_mut().enumerate() {
        match request.prepare(now, batch_wait) {
            Ok(true) => {
                if !ready.contains_key(&request.model) {
                    model_order.push(request.model.clone());
                }
                ready.entry(request.model.clone()).or_default().push(index);
            }
            Ok(false) => {}
            Err(error) => {
                let _ = request.output.send(Delivery::Error(error));
            }
        }
    }
    for indices in ready.values_mut() {
        indices.sort_by_key(|index| std::cmp::Reverse(requests[*index].priority(now)));
    }
    model_order.sort_by_key(|model| std::cmp::Reverse(ready[model].first().map_or(0, |index| requests[*index].priority(now))));
    for model_id in model_order {
        let indices = ready.get_mut(&model_id).expect("model came from ready map");
        let starts = indices.iter().filter(|index| requests[**index].active.is_none()).count();
        let capacity = registry.batch_capacity(&model_id).expect("ready model is loaded");
        let start_due = indices
            .iter()
            .filter(|index| requests[**index].active.is_none())
            .any(|index| requests[*index].batch_at.is_none_or(|deadline| deadline <= now));
        if starts == indices.len() && starts < capacity && !start_due {
            continue;
        }
        while !indices.is_empty() {
            let Some((position, (entry, model, max_batch))) = indices.iter().enumerate().find_map(|(position, index)| {
                let pinned = requests[*index].active.as_ref().map(|active| active.entry);
                registry.choose(&model_id, pinned).map(|chosen| (position, chosen))
            }) else {
                break;
            };
            let first = indices.remove(position);
            let operation = if requests[first].active.is_some() { ModelOperation::Continue } else { ModelOperation::Start };
            let mut selected = vec![first];
            let mut candidate = 0;
            while selected.len() < max_batch && candidate < indices.len() {
                let request = &requests[indices[candidate]];
                let request_operation = if request.active.is_some() { ModelOperation::Continue } else { ModelOperation::Start };
                let compatible_entry = request.active.as_ref().is_none_or(|active| active.entry == entry);
                if request_operation == operation && compatible_entry {
                    selected.push(indices.remove(candidate));
                } else {
                    candidate += 1;
                }
            }
            let mut batch = Vec::with_capacity(selected.len());
            let mut dispatches = Vec::with_capacity(selected.len());
            for index in selected {
                let request = &mut requests[index];
                if operation == ModelOperation::Start {
                    let chunk = request.prepared.pop_front().expect("ready request has prepared text");
                    request.committed = chunk.bytes;
                    request.committed_chars = chunk.span.end;
                    request.active = Some(ActiveChunk { entry, text: chunk.text, span: chunk.span });
                    request.text_index += 1;
                    request.start_time = Some(now);
                    request.batch_at = None;
                    request.collected_time = Duration::ZERO;
                    request.deadline = (request.committed < request.text.len()).then(|| now + request.params.timeout);
                }
                let active = request.active.as_ref().expect("ready request has active text");
                request.pending += 1;
                batch.push(ModelRequest {
                    stream_id: request.id,
                    operation,
                    text: active.text.clone(),
                    voice_id: request.voice.clone(),
                    params: request.params.model.clone(),
                    state: request.state.clone(),
                    alignment_type: request.params.alignment_type,
                });
                dispatches.push(Dispatch {
                    request_id: request.id,
                    chunk_index: request.output_index,
                    text_span: active.span.clone(),
                    nonce: request.nonce,
                });
            }
            work.send(Call { entry, model, batch, dispatches }).expect("caller pool exists while coordinator runs");
        }
    }
}

pub(crate) fn complete(registry: &mut Registry, requests: &mut [Request], call: Call, result: Result<Vec<ModelOutput>>) {
    registry.release(call.entry);
    let outputs = match result {
        Ok(outputs) if outputs.len() == call.dispatches.len() => outputs,
        Ok(_) => return fail(requests, &call, Error::Inference("model output count differs from batch".into())),
        Err(error) => return fail(requests, &call, error),
    };
    for (dispatch, output) in call.dispatches.into_iter().zip(outputs) {
        let Some(request) = requests.iter_mut().find(|request| request.id == dispatch.request_id && request.nonce == dispatch.nonce) else {
            continue;
        };
        request.pending -= 1;
        request.state = output.state;
        if !output.audio.is_empty() {
            request.collected_time += Duration::from_secs_f64(output.audio.len() as f64 / output.sample_rate as f64);
            let _ = request.output.send(Delivery::Chunk(crate::AudioChunk {
                audio: output.audio,
                sample_rate: output.sample_rate,
                chunk_index: dispatch.chunk_index,
                text_span: dispatch.text_span,
                alignment: output.alignment,
            }));
            request.output_index += 1;
        }
        if output.complete {
            request.active = None;
        }
        if output.complete && request.pending == 0 && request.committed == request.text.len() {
            let _ = request.output.send(Delivery::End(request.finished));
        }
    }
}

fn fail(requests: &mut [Request], call: &Call, error: Error) {
    for dispatch in &call.dispatches {
        if let Some(request) = requests.iter_mut().find(|request| request.id == dispatch.request_id && request.nonce == dispatch.nonce) {
            request.pending -= 1;
            request.collected_time = Duration::ZERO;
            request.start_time = None;
            request.active = None;
            let _ = call.model.close_stream(request.id);
            let _ = request.output.send(Delivery::Error(error.clone()));
        }
    }
}
