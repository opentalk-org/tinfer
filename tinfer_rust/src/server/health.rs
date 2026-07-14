use std::sync::Arc;
use std::sync::RwLock;
use std::sync::atomic::{AtomicUsize, Ordering};

use tokio::sync::Notify;

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum ServingState {
    #[default]
    Starting,
    Ready,
    Draining,
    Stopped,
}

#[derive(Default)]
pub struct HealthState {
    state: RwLock<ServingState>,
    active: AtomicUsize,
    changed: Notify,
}

pub(crate) struct Admission(Arc<HealthState>);

impl HealthState {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn state(&self) -> ServingState {
        *self.state.read().expect("health state lock poisoned")
    }

    pub fn ready(&self) -> bool {
        self.state() == ServingState::Ready
    }

    pub fn active_admissions(&self) -> usize {
        self.active.load(Ordering::Acquire)
    }

    pub(crate) fn set(&self, state: ServingState) {
        *self.state.write().expect("health state lock poisoned") = state;
        self.changed.notify_waiters();
    }

    pub(crate) fn admit(self: &Arc<Self>) -> Option<Admission> {
        if !self.ready() {
            return None;
        }
        self.active.fetch_add(1, Ordering::AcqRel);
        if self.ready() {
            Some(Admission(self.clone()))
        } else {
            self.active.fetch_sub(1, Ordering::AcqRel);
            None
        }
    }

    pub(crate) async fn drained(&self) {
        while self.active_admissions() != 0 {
            self.changed.notified().await;
        }
    }
}

impl Drop for Admission {
    fn drop(&mut self) {
        self.0.active.fetch_sub(1, Ordering::AcqRel);
        self.0.changed.notify_waiters();
    }
}
