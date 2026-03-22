//! Read-only status projections for the worker pool.
//!
//! These methods query pool state without mutating it. They use `try_lock()`
//! and `Ordering::Relaxed` — never blocking, never acquiring exclusive access.

use std::sync::atomic::Ordering;

use crate::api::{ReleasedCommand, WorkerLanguage};
use crate::worker::WorkerProfile;

use super::{WorkerPool, lock_recovered};

impl WorkerPool {
    /// Check if there are idle workers for a given `(command, lang)` key.
    ///
    /// Used by the memory gate to skip the system memory check when reusable
    /// workers already exist -- those workers are already loaded, so no new
    /// memory allocation is needed. Checks both stdio and TCP workers.
    pub fn has_idle_workers(&self, command: ReleasedCommand, lang: impl Into<WorkerLanguage>) -> bool {
        let lang = lang.into();
        let Some(profile) = WorkerProfile::for_command(command) else {
            return false;
        };

        // GPU profile workers are always "available" (shared, concurrent).
        if profile.is_concurrent() {
            // Check TCP GPU workers first.
            if let Ok(tcp_gpu_workers) = self.gpu_tcp_workers.try_lock()
                && tcp_gpu_workers.keys().any(|(l, _)| l == &lang)
            {
                return true;
            }
            let gpu_workers = self.gpu_workers.try_lock().ok();
            if let Some(gpu_workers) = gpu_workers {
                return gpu_workers.keys().any(|(l, _)| l == &lang);
            }
            return false;
        }

        let groups = lock_recovered(&self.groups);
        groups
            .iter()
            .any(|((group_profile, group_lang, _), group)| {
                if *group_profile != profile || group_lang != &lang {
                    return false;
                }
                !lock_recovered(&group.idle).is_empty()
                    || !lock_recovered(&group.tcp_workers).is_empty()
            })
    }

    /// Number of active workers (total across all keys, including checked-out).
    ///
    /// Counts sequential group workers, shared GPU workers, and TCP workers.
    pub fn worker_count(&self) -> usize {
        let groups_count: usize = {
            let groups = lock_recovered(&self.groups);
            groups
                .values()
                .map(|g| g.total.load(Ordering::Relaxed))
                .sum()
        };
        let gpu_count = self.gpu_workers.try_lock().map(|g| g.len()).unwrap_or(0);
        let tcp_gpu_count = self
            .gpu_tcp_workers
            .try_lock()
            .map(|g| g.len())
            .unwrap_or(0);
        groups_count + gpu_count + tcp_gpu_count
    }

    /// Active worker keys: `["profile:stanza:eng (2 total, 1 idle)", ...]`.
    ///
    /// Includes both sequential group workers and shared GPU workers.
    pub fn worker_keys(&self) -> Vec<String> {
        let groups = lock_recovered(&self.groups);
        let mut keys: Vec<String> = groups
            .iter()
            .map(|((profile, lang, engine_overrides), group)| {
                let total = group.total.load(Ordering::Relaxed);
                let idle = lock_recovered(&group.idle).len();
                let suffix = if engine_overrides.is_empty() {
                    String::new()
                } else {
                    format!(":{}", engine_overrides)
                };
                format!(
                    "{}:{lang}{suffix} ({total} total, {idle} idle)",
                    profile.label()
                )
            })
            .collect();
        drop(groups);

        if let Ok(gpu_workers) = self.gpu_workers.try_lock() {
            for ((lang, engine_overrides), _worker) in gpu_workers.iter() {
                let suffix = if engine_overrides.is_empty() {
                    String::new()
                } else {
                    format!(":{}", engine_overrides)
                };
                keys.push(format!("profile:gpu:{lang}{suffix} (1 total, shared)"));
            }
        }

        if let Ok(tcp_gpu_workers) = self.gpu_tcp_workers.try_lock() {
            for ((lang, engine_overrides), _worker) in tcp_gpu_workers.iter() {
                let suffix = if engine_overrides.is_empty() {
                    String::new()
                } else {
                    format!(":{}", engine_overrides)
                };
                keys.push(format!("profile:gpu:{lang}{suffix} (1 total, tcp-shared)"));
            }
        }

        keys.sort();
        keys
    }

    /// Summary of idle workers: `["profile:stanza:eng:pid=1234:transport=stdio", ...]`.
    ///
    /// Reports idle sequential workers and shared GPU workers. Checked-out
    /// sequential workers are invisible; use `worker_count()` for full totals.
    pub fn worker_summary(&self) -> Vec<String> {
        let groups = lock_recovered(&self.groups);
        let mut summary = Vec::new();
        for group in groups.values() {
            let idle = lock_recovered(&group.idle);
            for worker in idle.iter() {
                summary.push(format!(
                    "{}:{}:pid={}:transport={}",
                    worker.profile_label(),
                    worker.lang(),
                    worker.pid(),
                    worker.transport()
                ));
            }
        }
        drop(groups);

        if let Ok(gpu_workers) = self.gpu_workers.try_lock() {
            for ((_lang, _engine_overrides), worker) in gpu_workers.iter() {
                summary.push(format!(
                    "{}:{}:pid={}:transport=stdio:concurrent",
                    worker.profile_label(),
                    worker.lang(),
                    worker.pid()
                ));
            }
        }

        if let Ok(tcp_gpu_workers) = self.gpu_tcp_workers.try_lock() {
            for ((lang, _engine_overrides), worker) in tcp_gpu_workers.iter() {
                summary.push(format!(
                    "profile:gpu:{}:pid={}:transport=tcp:concurrent",
                    lang,
                    worker.pid()
                ));
            }
        }

        // Include TCP workers from sequential groups.
        {
            let groups = lock_recovered(&self.groups);
            for group in groups.values() {
                let tcp = lock_recovered(&group.tcp_workers);
                for worker in tcp.iter() {
                    summary.push(format!(
                        "{}:{}:pid={}:transport=tcp",
                        worker.profile_label(),
                        worker.lang(),
                        worker.pid()
                    ));
                }
            }
        }

        summary.sort();
        summary
    }
}
