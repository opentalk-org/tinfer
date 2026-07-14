use std::ops::Range;
use std::sync::Arc;
use std::thread::JoinHandle;

use crossbeam_channel::{Receiver, Sender};

use super::engine::Message;
use super::registry::EntryId;
use crate::models::Model;
use crate::ModelRequest;

#[derive(Clone)]
pub(crate) struct Dispatch {
    pub request_id: u64,
    pub chunk_index: u64,
    pub text_span: Range<usize>,
    pub nonce: u64,
}

#[derive(Clone)]
pub(crate) struct Call {
    pub entry: EntryId,
    pub model: Arc<dyn Model>,
    pub batch: Vec<ModelRequest>,
    pub dispatches: Vec<Dispatch>,
}

pub(crate) fn callers(count: usize, rx: Receiver<Call>, tx: Sender<Message>) -> Vec<JoinHandle<()>> {
    (0..count)
        .map(|_| {
            let rx = rx.clone();
            let tx = tx.clone();
            std::thread::spawn(move || {
                while let Ok(call) = rx.recv() {
                    let result = call.model.generate_batch(&call.batch);
                    if tx.send(Message::Complete(call, result)).is_err() {
                        break;
                    }
                }
            })
        })
        .collect()
}
