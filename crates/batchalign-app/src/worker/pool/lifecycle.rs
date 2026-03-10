//! Background tasks: health checking, idle timeout, worker spawning helpers.

use std::sync::Arc;
use std::sync::atomic::Ordering;
use std::time::Duration;

use crate::api::{LanguageCode3, NumSpeakers};
use tracing::{error, info, warn};

use crate::worker::WorkerTarget;
use crate::worker::error::WorkerError;
use crate::worker::handle::{WorkerConfig, WorkerHandle};

use super::{GroupsMap, WorkerGroup, WorkerKey, WorkerPool};

impl WorkerPool {
    /// Start background tasks for health checking and idle timeout.
    ///
    /// Returns a `JoinHandle` that completes when the pool is shut down.
    pub fn start_background_tasks(&self) -> tokio::task::JoinHandle<()> {
        let groups = self.groups.clone();
        let cancel = self.cancel.clone();
        let health_interval = Duration::from_secs(self.config.health_check_interval_s);
        let idle_timeout = Duration::from_secs(self.config.idle_timeout_s);
        let pool_config = self.config.clone();

        tokio::spawn(async move {
            let mut interval = tokio::time::interval(health_interval);
            interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);

            loop {
                tokio::select! {
                    _ = cancel.cancelled() => {
                        info!("Worker pool background tasks cancelled");
                        break;
                    }
                    _ = interval.tick() => {
                        run_health_check(
                            &groups, idle_timeout, &pool_config,
                        ).await;
                    }
                }
            }
        })
    }

    /// Build a `WorkerConfig` for the given worker target and language.
    pub(super) fn worker_config(
        &self,
        target: &WorkerTarget,
        lang: &LanguageCode3,
        engine_overrides: &str,
    ) -> WorkerConfig {
        WorkerConfig {
            python_path: self.config.python_path.clone(),
            target: *target,
            lang: lang.clone(),
            num_speakers: NumSpeakers(1),
            engine_overrides: engine_overrides.to_owned(),
            test_echo: self.config.test_echo,
            ready_timeout_s: self.config.ready_timeout_s,
            verbose: self.config.verbose,
            runtime: self.config.runtime.clone(),
        }
    }

    /// Get or create the `WorkerGroup` for a key.
    pub(super) fn get_or_create_group(
        &self,
        target: &WorkerTarget,
        lang: &LanguageCode3,
        engine_overrides: &str,
    ) -> Arc<WorkerGroup> {
        let key: super::WorkerKey = (*target, lang.clone(), engine_overrides.to_owned());
        let mut groups = self.groups.lock().unwrap();
        groups
            .entry(key)
            .or_insert_with(|| Arc::new(WorkerGroup::new()))
            .clone()
    }

    /// Try to atomically claim a spawn slot in a group via compare_exchange.
    ///
    /// Returns `Ok(claimed_total)` if a slot was claimed, `Err(current)` if
    /// the group is already at capacity.
    pub(super) fn try_claim_spawn_slot(&self, group: &WorkerGroup) -> Result<usize, usize> {
        let max = self.config.max_workers_per_key;
        loop {
            let current = group.total.load(Ordering::Relaxed);
            if current >= max {
                return Err(current);
            }
            match group.total.compare_exchange(
                current,
                current + 1,
                Ordering::Relaxed,
                Ordering::Relaxed,
            ) {
                Ok(_) => return Ok(current + 1),
                Err(_) => continue, // Retry CAS
            }
        }
    }

    /// Spawn a worker into a group, using `try_claim_spawn_slot` for the
    /// atomic slot reservation.
    ///
    /// On success, the worker is pushed into the idle queue with a permit.
    /// On spawn failure, the slot is released.
    pub(super) async fn try_spawn_into_group(
        &self,
        group: &Arc<WorkerGroup>,
        target: &WorkerTarget,
        lang: &LanguageCode3,
        engine_overrides: &str,
    ) -> Result<bool, WorkerError> {
        if self.try_claim_spawn_slot(group).is_err() {
            return Ok(false); // At capacity
        }

        let _bootstrap_guard = group.bootstrap.lock().await;

        // Slot claimed -- now spawn. If spawn fails, release the slot.
        match WorkerHandle::spawn(self.worker_config(target, lang, engine_overrides)).await {
            Ok(handle) => {
                // Don't use a separate push_spawned (which would double-increment
                // total). We already incremented via compare_exchange.
                group.idle.lock().unwrap().push_back(handle);
                group.available.add_permits(1);
                Ok(true)
            }
            Err(e) => {
                // Release the slot we claimed
                group.total.fetch_sub(1, Ordering::Relaxed);
                Err(e)
            }
        }
    }
}

/// Run a single round of health checks and idle timeout enforcement.
///
/// Only examines idle workers (checked-out workers are in use -- errors
/// during dispatch are handled by the caller). Dead or timed-out workers
/// are removed from the idle queue and `total` is decremented.
pub(super) async fn run_health_check(
    groups_ref: &GroupsMap,
    idle_timeout: Duration,
    pool_config: &super::PoolConfig,
) {
    // Snapshot group Arcs so we don't hold the groups lock across awaits.
    let group_snapshot: Vec<(WorkerKey, Arc<WorkerGroup>)> = {
        let groups = groups_ref.lock().unwrap();
        groups.iter().map(|(k, g)| (k.clone(), g.clone())).collect()
    };

    for (key, group) in &group_snapshot {
        // Drain the idle queue for health checking.
        let workers_to_check: Vec<WorkerHandle> =
            { group.idle.lock().unwrap().drain(..).collect() };
        // We drained idle workers. Their permits are already consumed
        // (no one can acquire them). We'll re-add permits for healthy ones.

        let mut to_return = Vec::new();
        let mut restart_count = 0usize;
        let mut removed_count = 0usize;

        for mut worker in workers_to_check {
            // Check idle timeout
            if worker.idle_duration() > idle_timeout {
                info!(
                    target = %key.0.label(),
                    lang = %key.1,
                    engine_overrides = %key.2,
                    pid = %worker.pid(),
                    idle_s = worker.idle_duration().as_secs(),
                    "Worker idle timeout, shutting down"
                );
                let _ = worker.shutdown_in_place().await;
                removed_count += 1;
                continue;
            }

            // Check if process is alive
            if !worker.is_alive() {
                warn!(
                    target = %key.0.label(),
                    lang = %key.1,
                    engine_overrides = %key.2,
                    pid = %worker.pid(),
                    "Worker process died, scheduling restart"
                );
                removed_count += 1;
                restart_count += 1;
                // worker dropped here (SIGTERM+SIGKILL via WorkerHandle::Drop)
                continue;
            }

            // Health check via worker IPC
            match worker.health_check().await {
                Ok(_) => {
                    to_return.push(worker);
                }
                Err(e) => {
                    warn!(
                        target = %key.0.label(),
                        lang = %key.1,
                        engine_overrides = %key.2,
                        pid = %worker.pid(),
                        error = %e,
                        "Health check failed, scheduling restart"
                    );
                    removed_count += 1;
                    restart_count += 1;
                }
            }
        }

        // Return healthy workers
        {
            let returned = to_return.len();
            let mut idle = group.idle.lock().unwrap();
            for w in to_return {
                idle.push_back(w);
            }
            group.available.add_permits(returned);
        }

        // Decrement total for removed workers
        if removed_count > 0 {
            group.total.fetch_sub(removed_count, Ordering::Relaxed);
        }

        // Restart failed workers
        for _ in 0..restart_count {
            info!(
                target = %key.0.label(),
                lang = %key.1,
                engine_overrides = %key.2,
                "Restarting worker"
            );

            let _bootstrap_guard = group.bootstrap.lock().await;

            let config = WorkerConfig {
                python_path: pool_config.python_path.clone(),
                target: key.0,
                lang: key.1.clone(),
                num_speakers: NumSpeakers(1),
                engine_overrides: key.2.clone(),
                test_echo: pool_config.test_echo,
                ready_timeout_s: pool_config.ready_timeout_s,
                verbose: pool_config.verbose,
                runtime: pool_config.runtime.clone(),
            };

            match WorkerHandle::spawn(config).await {
                Ok(handle) => {
                    let pid = handle.pid();
                    group.total.fetch_add(1, Ordering::Relaxed);
                    group.idle.lock().unwrap().push_back(handle);
                    group.available.add_permits(1);
                    info!(
                        target = %key.0.label(),
                        lang = %key.1,
                        engine_overrides = %key.2,
                        pid = %pid,
                        "Worker restarted"
                    );
                }
                Err(e) => {
                    error!(
                        target = %key.0.label(),
                        lang = %key.1,
                        engine_overrides = %key.2,
                        error = %e,
                        "Failed to restart worker"
                    );
                }
            }
        }
    }

    // Clean up empty groups
    {
        let mut groups = groups_ref.lock().unwrap();
        groups.retain(|_, g| g.total.load(Ordering::Relaxed) > 0);
    }
}
