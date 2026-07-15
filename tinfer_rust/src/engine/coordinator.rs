use std::time::Duration;

use crossbeam_channel::{Receiver, RecvTimeoutError, Sender, bounded};

use super::caller::Call;
use super::engine::{Delivery, LoadedModel, Message};
use super::registry::Registry;
use super::scheduler::{Request, dispatch, find_request, next_wait, unload};
use crate::Error;

pub(super) fn coordinate(loaded: Vec<LoadedModel>, rx: Receiver<Message>, work: Sender<Call>, batch_wait: Duration) {
    let mut registry = Registry::default();
    for (model, max_batch, settings) in loaded {
        registry.add(model, max_batch, settings);
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
        dispatch(&mut registry, &mut requests, &work, batch_wait);
        if stop {
            return;
        }
    }
}

fn process(message: Message, registry: &mut Registry, requests: &mut Vec<Request>, next_id: &mut u64) -> bool {
    match message {
        Message::Load(config, reply) => {
            let result = config
                .settings
                .as_object()
                .ok_or_else(|| Error::Validation("model settings must be an object".into()))
                .and_then(|settings| crate::models::load(&config).map(|model| (model, settings.clone())))
                .map(|(model, settings)| registry.add(model, config.max_batch, settings));
            let _ = reply.send(result);
        }
        Message::Unload(model, reply) => {
            let result = requests
                .iter()
                .filter(|request| request.model == model)
                .try_for_each(|request| registry.close_stream(&model, request.id))
                .and_then(|()| registry.unload(&model));
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
        Message::Create(model, voice, mut params, text, reply) => {
            let result = registry.validate(&model, &voice).and_then(|default_language| {
                let overrides =
                    params.model.as_object().ok_or_else(|| Error::Validation("model request settings must be an object".into()))?;
                let mut settings = registry.settings(&model)?;
                settings.extend(overrides.clone());
                params.model = serde_json::Value::Object(settings);
                let language =
                    params.model["language"].as_str().filter(|language| !language.is_empty()).unwrap_or(&default_language).to_owned();
                params.model["language"] = serde_json::Value::String(language.clone());
                let id = *next_id;
                *next_id += 1;
                let (tx, rx) = bounded(2);
                let mut request = Request::new(id, model, voice, language, params, tx)?;
                if let Some(text) = text {
                    request.append(text);
                    request.forced = true;
                }
                requests.push(request);
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
            let result = find_request(requests, id).and_then(|request| {
                registry.close_stream(&request.model, request.id)?;
                request.cancelled = true;
                request.text.clear();
                request.committed = 0;
                request.committed_chars = 0;
                request.prepared.clear();
                request.active = None;
                request.deadline = None;
                request.batch_at = None;
                request.forced = false;
                request.pending = 0;
                request.nonce = request.nonce.wrapping_add(1);
                let _ = request.output.send(Delivery::Error(Error::Cancelled));
                Ok(())
            });
            let _ = reply.send(result);
        }
        Message::State(id, reply) => {
            let _ = reply.send(find_request(requests, id).map(|request| request.state.clone()));
        }
        Message::Close(id, reply) => {
            let result = find_request(requests, id).and_then(|request| registry.close_stream(&request.model, request.id));
            if result.is_ok() {
                requests.retain(|request| request.id != id);
            }
            let _ = reply.send(result);
        }
        Message::Complete(call, result) => super::scheduler::complete(registry, requests, call, result),
        Message::Stop(reply) => {
            let mut result = Ok(());
            for request in requests.drain(..) {
                if let Err(error) = registry.close_stream(&request.model, request.id) {
                    result = Err(error);
                }
                let _ = request.output.send(Delivery::Error(Error::Shutdown));
            }
            let _ = reply.send(result);
            return true;
        }
    }
    false
}
