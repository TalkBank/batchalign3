//! Queue backend abstraction and local dispatcher.
//!
//! This module defines the first explicit queue/backend boundary in the Rust
//! server. Routes and lifecycle handlers interact with a [`QueueBackend`]
//! instead of a concrete dispatcher implementation, while the local server
//! continues to use an in-process backend plus dispatcher loop.

use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use tokio::sync::Notify;
use tracing::info;

use crate::api::{JobId, UnixTimestamp};
use crate::runner::{RunnerContext, job_task};
use crate::runtime_supervisor::RuntimeSupervisor;
use crate::store::{JobStore, unix_now};

const IDLE_POLL_INTERVAL: Duration = Duration::from_secs(30);

fn wait_duration(now: UnixTimestamp, next_wake_at: Option<UnixTimestamp>) -> Duration {
    match next_wake_at {
        Some(ts) if ts > now => Duration::from_secs_f64(ts.0 - now.0),
        Some(_) => Duration::ZERO,
        None => IDLE_POLL_INTERVAL,
    }
}

/// Result of polling a queue backend for currently eligible queued jobs.
#[derive(Debug, Default)]
pub struct QueuePoll {
    /// Job IDs that are ready to run now and have been claimed by the backend.
    pub ready_job_ids: Vec<JobId>,
    /// Earliest future eligibility timestamp among still-queued jobs.
    pub next_wake_at: Option<UnixTimestamp>,
}

/// Abstract queue/backend interface for queued job dispatch.
#[async_trait]
pub trait QueueBackend: Send + Sync {
    /// Notify the backend that queued-job state may have changed.
    fn notify(&self);

    /// Claim queued jobs that are currently eligible to run.
    async fn claim_ready_jobs(&self) -> QueuePoll;

    /// Wait until queue state changes or the next eligibility deadline arrives.
    async fn wait_for_work(&self, next_wake_at: Option<UnixTimestamp>);
}

/// In-process queue backend backed by [`JobStore`] state.
///
/// This backend claims eligible queued jobs from the in-memory store and uses
/// a local [`Notify`] to wake the dispatcher when queued-job state changes.
pub struct LocalQueueBackend {
    store: Arc<JobStore>,
    notify: Arc<Notify>,
}

impl LocalQueueBackend {
    /// Create a new local queue backend.
    pub fn new(store: Arc<JobStore>, notify: Arc<Notify>) -> Self {
        Self { store, notify }
    }
}

#[async_trait]
impl QueueBackend for LocalQueueBackend {
    fn notify(&self) {
        self.notify.notify_one();
    }

    async fn claim_ready_jobs(&self) -> QueuePoll {
        self.store.claim_ready_queued_jobs().await
    }

    async fn wait_for_work(&self, next_wake_at: Option<UnixTimestamp>) {
        let delay = wait_duration(unix_now(), next_wake_at);
        tokio::select! {
            _ = self.notify.notified() => {}
            _ = tokio::time::sleep(delay) => {}
        }
    }
}

/// Local dispatcher that launches runner tasks from a [`QueueBackend`].
///
/// The dispatcher owns the transition from claimed queued jobs to active runner
/// tasks. This keeps queue-state decisions in the backend and execution launch
/// logic in a separate host-side component.
pub struct QueueDispatcher {
    backend: Arc<dyn QueueBackend>,
    supervisor: RuntimeSupervisor,
    runner_context: RunnerContext,
}

impl QueueDispatcher {
    /// Create a new local queue dispatcher.
    pub fn new(
        backend: Arc<dyn QueueBackend>,
        supervisor: RuntimeSupervisor,
        runner_context: RunnerContext,
    ) -> Self {
        Self {
            backend,
            supervisor,
            runner_context,
        }
    }

    /// Run the dispatcher loop until process shutdown.
    pub async fn run(self) {
        loop {
            let poll = self.backend.claim_ready_jobs().await;

            if !poll.ready_job_ids.is_empty() {
                info!(count = poll.ready_job_ids.len(), "Dispatching queued jobs");
                for job_id in poll.ready_job_ids {
                    self.supervisor
                        .spawn_job(job_task(job_id, self.runner_context.clone()));
                }
                continue;
            }

            self.backend.wait_for_work(poll.next_wake_at).await;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// When there is no queued-job deadline, the dispatcher sleeps for the idle poll interval.
    #[test]
    fn wait_duration_uses_idle_poll_interval_without_deadline() {
        assert_eq!(wait_duration(UnixTimestamp(10.0), None), IDLE_POLL_INTERVAL);
    }

    /// Expired deadlines should wake the dispatcher immediately.
    #[test]
    fn wait_duration_is_zero_for_expired_deadline() {
        assert_eq!(
            wait_duration(UnixTimestamp(20.0), Some(UnixTimestamp(19.0))),
            Duration::ZERO
        );
    }
}
