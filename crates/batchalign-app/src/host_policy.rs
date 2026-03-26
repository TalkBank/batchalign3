//! Host-level execution policy resolved once from runtime configuration.
//!
//! Command metadata says what a command *can* do. The host policy says what this
//! machine should actually allow right now: how many jobs to admit, whether
//! startup warmup is appropriate, and whether local workers should bootstrap an
//! entire shared profile or only one task to minimize resident memory.

use crate::commands::spec::{ConstrainedHostPolicy, WarmupPolicy};
use crate::config::ServerConfig;
use crate::runtime::{MemoryTier, MemoryTierKind};
use crate::worker::WorkerBootstrapMode;

/// Hard upper bound on auto-tuned concurrent job slots.
pub const AUTO_CONCURRENT_MAX_SLOTS: usize = 8;

/// Fallback slot count when the CPU count cannot be detected.
pub const AUTO_CONCURRENT_FALLBACK_SLOTS: usize = 4;

/// Host decision for startup warmup.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StartupWarmupPolicy {
    /// Keep startup fully lazy.
    Disabled,
    /// Allow warmup for commands whose metadata marks them eligible.
    EligibleCommands,
}

/// Resolved host-level execution policy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HostExecutionPolicy {
    /// Memory tier resolved for this host.
    pub memory_tier: MemoryTier,
    /// How local worker processes should bootstrap.
    pub bootstrap_mode: WorkerBootstrapMode,
    /// Whether startup warmup is allowed at all.
    pub startup_warmup: StartupWarmupPolicy,
}

impl Default for HostExecutionPolicy {
    fn default() -> Self {
        Self::from_memory_tier(MemoryTier::from_total_mb(64_000))
    }
}

impl HostExecutionPolicy {
    /// Resolve host policy from one concrete memory tier.
    pub fn from_memory_tier(memory_tier: MemoryTier) -> Self {
        let constrained = matches!(memory_tier.kind, MemoryTierKind::Small);
        Self {
            memory_tier,
            bootstrap_mode: if constrained {
                WorkerBootstrapMode::Task
            } else {
                WorkerBootstrapMode::Profile
            },
            startup_warmup: if constrained {
                StartupWarmupPolicy::Disabled
            } else {
                StartupWarmupPolicy::EligibleCommands
            },
        }
    }

    /// Resolve host policy from the validated server configuration.
    pub fn from_server_config(config: &ServerConfig) -> Self {
        Self::from_memory_tier(config.resolved_memory_tier())
    }

    /// Whether this host should take the constrained-memory path.
    pub fn is_constrained_host(self) -> bool {
        self.bootstrap_mode == WorkerBootstrapMode::Task
    }

    /// Resolve auto job concurrency using the current host policy.
    pub fn auto_max_concurrent_jobs(self) -> usize {
        let by_cpu = std::thread::available_parallelism()
            .map(|parallelism| parallelism.get())
            .unwrap_or(AUTO_CONCURRENT_FALLBACK_SLOTS);
        auto_max_concurrent_from(by_cpu, self.memory_tier.max_suggested_workers)
    }

    /// Resolve the per-command file-parallelism hint under current host limits.
    pub(crate) fn resolved_file_parallelism(
        self,
        constrained_host: ConstrainedHostPolicy,
        suggested_parallelism: usize,
    ) -> usize {
        if self.is_constrained_host()
            && matches!(
                constrained_host,
                ConstrainedHostPolicy::SequentialFallback
                    | ConstrainedHostPolicy::DelegatedToSubcommands
            )
        {
            1
        } else {
            suggested_parallelism.max(1)
        }
    }

    /// Whether one command remains eligible for startup warmup on this host.
    pub(crate) fn allows_command_warmup(self, warmup: WarmupPolicy, test_echo_mode: bool) -> bool {
        test_echo_mode
            && matches!(self.startup_warmup, StartupWarmupPolicy::EligibleCommands)
            && matches!(warmup, WarmupPolicy::BackgroundEligible)
    }
}

/// Resolve auto job concurrency from CPU and memory caps.
pub fn auto_max_concurrent_from(by_cpu: usize, by_memory: usize) -> usize {
    by_cpu
        .clamp(1, AUTO_CONCURRENT_MAX_SLOTS)
        .min(by_memory.max(1))
}

#[cfg(test)]
mod tests {
    use super::{HostExecutionPolicy, StartupWarmupPolicy};
    use crate::commands::spec::{ConstrainedHostPolicy, WarmupPolicy};
    use crate::runtime::MemoryTier;
    use crate::worker::WorkerBootstrapMode;

    #[test]
    fn small_hosts_choose_task_bootstrap_and_disable_warmup() {
        let policy = HostExecutionPolicy::from_memory_tier(MemoryTier::from_total_mb(16_000));
        assert_eq!(policy.bootstrap_mode, WorkerBootstrapMode::Task);
        assert_eq!(policy.startup_warmup, StartupWarmupPolicy::Disabled);
        assert!(policy.is_constrained_host());
    }

    #[test]
    fn large_hosts_keep_profile_bootstrap() {
        let policy = HostExecutionPolicy::from_memory_tier(MemoryTier::from_total_mb(64_000));
        assert_eq!(policy.bootstrap_mode, WorkerBootstrapMode::Profile);
        assert_eq!(policy.startup_warmup, StartupWarmupPolicy::EligibleCommands);
        assert!(!policy.is_constrained_host());
    }

    #[test]
    fn constrained_hosts_clamp_sequential_fallback_commands() {
        let policy = HostExecutionPolicy::from_memory_tier(MemoryTier::from_total_mb(16_000));
        assert_eq!(
            policy.resolved_file_parallelism(ConstrainedHostPolicy::SequentialFallback, 4),
            1
        );
        assert_eq!(
            policy.resolved_file_parallelism(ConstrainedHostPolicy::DelegatedToSubcommands, 4),
            1
        );
    }

    #[test]
    fn warmup_requires_host_and_command_eligibility() {
        let policy = HostExecutionPolicy::from_memory_tier(MemoryTier::from_total_mb(64_000));
        assert!(policy.allows_command_warmup(WarmupPolicy::BackgroundEligible, true));
        assert!(!policy.allows_command_warmup(WarmupPolicy::DelegatedToSubcommands, true));
        assert!(!policy.allows_command_warmup(WarmupPolicy::BackgroundEligible, false));
    }

    #[test]
    fn auto_job_slots_use_memory_cap() {
        let policy = HostExecutionPolicy::from_memory_tier(MemoryTier::from_total_mb(16_000));
        assert_eq!(policy.auto_max_concurrent_jobs(), 1);
    }
}
