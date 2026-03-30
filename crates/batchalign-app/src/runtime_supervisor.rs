//! Explicit owner for background runtime tasks.
//!
//! The server has two long-lived categories of background work:
//!
//! - the queue dispatcher loop
//! - per-job runner tasks
//!
//! This module keeps those tasks behind one owned actor instead of exposing
//! shared `Mutex<JoinSet<_>>` and `Mutex<Option<JoinHandle<_>>>` fields across
//! the application state.

use std::future::Future;
use std::pin::Pin;
use std::time::Duration;

use tokio::sync::mpsc::{UnboundedReceiver, UnboundedSender, unbounded_channel};
use tokio::sync::oneshot;
use tokio::task::JoinSet;

/// Heap-allocated background task future accepted by the supervisor.
type BackgroundTask = Pin<Box<dyn Future<Output = ()> + Send + 'static>>;

/// Summary of the runtime supervisor shutdown sequence.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ShutdownSummary {
    /// True when the shutdown wait hit its deadline before every job task
    /// finished naturally.
    pub timed_out: bool,
    /// Number of job tasks still present in the supervisor when the deadline
    /// expired.
    pub remaining_jobs: usize,
}

/// Error returned when the runtime supervisor cannot report shutdown status.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum ShutdownError {
    /// The supervisor actor was already gone before the shutdown command could be sent.
    #[error("runtime supervisor unavailable before shutdown command could be sent")]
    Unavailable,
    /// The supervisor accepted shutdown but dropped the reply channel before responding.
    #[error("runtime supervisor dropped shutdown status before replying")]
    ReplyDropped,
}

/// Cloneable handle for the runtime supervisor actor.
///
/// Clones are cheap and all send commands into the same owned supervisor task.
#[derive(Clone)]
pub struct RuntimeSupervisor {
    commands: UnboundedSender<SupervisorCommand>,
}

/// Command sent to the runtime supervisor actor.
enum SupervisorCommand {
    /// Spawn one tracked per-job background task.
    SpawnJob {
        /// Future that owns the complete job lifecycle.
        task: BackgroundTask,
    },
    /// Stop the queue loop and wait for tracked jobs to finish.
    Shutdown {
        /// Maximum time to wait for job tasks before returning.
        timeout: Duration,
        /// Channel used to send the shutdown summary back to the caller.
        reply: oneshot::Sender<ShutdownSummary>,
    },
}

impl RuntimeSupervisor {
    /// Create and start a new runtime supervisor actor.
    pub fn new() -> Self {
        let (commands, receiver) = unbounded_channel();
        tokio::spawn(run_supervisor(receiver));
        Self { commands }
    }

    /// Spawn one tracked per-job background task.
    pub fn spawn_job<F>(&self, task: F)
    where
        F: Future<Output = ()> + Send + 'static,
    {
        let _ = self.commands.send(SupervisorCommand::SpawnJob {
            task: Box::pin(task),
        });
    }

    /// Stop the queue task and wait for tracked jobs to finish.
    pub async fn shutdown(&self, timeout: Duration) -> Result<ShutdownSummary, ShutdownError> {
        let (reply, receiver) = oneshot::channel();
        if self
            .commands
            .send(SupervisorCommand::Shutdown { timeout, reply })
            .is_err()
        {
            return Err(ShutdownError::Unavailable);
        }

        receiver.await.map_err(|_| ShutdownError::ReplyDropped)
    }
}

/// Run the task-supervisor actor loop.
async fn run_supervisor(mut receiver: UnboundedReceiver<SupervisorCommand>) {
    let mut queue_task: Option<tokio::task::JoinHandle<()>> = None;
    let mut job_tasks = JoinSet::new();

    while let Some(command) = receiver.recv().await {
        match command {
            SupervisorCommand::SpawnJob { task } => {
                job_tasks.spawn(task);
            }
            SupervisorCommand::Shutdown { timeout, reply } => {
                if let Some(handle) = queue_task.take() {
                    handle.abort();
                }

                let timed_out = tokio::time::timeout(timeout, async {
                    while job_tasks.join_next().await.is_some() {}
                })
                .await
                .is_err();
                let remaining_jobs = job_tasks.len();
                let _ = reply.send(ShutdownSummary {
                    timed_out,
                    remaining_jobs,
                });
                break;
            }
        }
    }

    if let Some(handle) = queue_task.take() {
        handle.abort();
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::time::Duration;

    use super::*;

    /// Shutdown waits for tracked job tasks when they complete before the deadline.
    #[tokio::test]
    async fn shutdown_waits_for_jobs() {
        let supervisor = RuntimeSupervisor::new();
        let completed = Arc::new(AtomicUsize::new(0));
        let completed_for_task = completed.clone();

        supervisor.spawn_job(async move {
            tokio::time::sleep(Duration::from_millis(10)).await;
            completed_for_task.store(1, Ordering::SeqCst);
        });

        let summary = supervisor
            .shutdown(Duration::from_secs(1))
            .await
            .expect("shutdown should succeed");

        assert!(!summary.timed_out);
        assert_eq!(summary.remaining_jobs, 0);
        assert_eq!(completed.load(Ordering::SeqCst), 1);
    }

    /// Shutdown reports timeout when a tracked job exceeds the deadline.
    #[tokio::test]
    async fn shutdown_reports_timed_out_jobs() {
        let supervisor = RuntimeSupervisor::new();

        supervisor.spawn_job(async {
            tokio::time::sleep(Duration::from_secs(60)).await;
        });

        let summary = supervisor
            .shutdown(Duration::from_millis(10))
            .await
            .expect("shutdown should succeed");

        assert!(summary.timed_out);
        assert!(summary.remaining_jobs >= 1);
    }

    /// Shutdown reports an explicit error instead of fabricating a clean summary
    /// when the supervisor actor is already unavailable.
    #[tokio::test]
    async fn shutdown_reports_unavailable_supervisor() {
        let (commands, receiver) = unbounded_channel();
        drop(receiver);
        let supervisor = RuntimeSupervisor { commands };

        let error = supervisor
            .shutdown(Duration::from_secs(1))
            .await
            .expect_err("shutdown should fail when supervisor is unavailable");

        assert_eq!(error, ShutdownError::Unavailable);
    }

    /// Shutdown reports an explicit error when the supervisor drops the reply
    /// channel before sending a summary.
    #[tokio::test]
    async fn shutdown_reports_dropped_reply() {
        let (commands, mut receiver) = unbounded_channel();
        tokio::spawn(async move {
            if let Some(SupervisorCommand::Shutdown { .. }) = receiver.recv().await {
                // Drop the reply without sending a summary.
            }
        });
        let supervisor = RuntimeSupervisor { commands };

        let error = supervisor
            .shutdown(Duration::from_secs(1))
            .await
            .expect_err("shutdown should fail when reply is dropped");

        assert_eq!(error, ShutdownError::ReplyDropped);
    }
}
