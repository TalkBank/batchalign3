//! Compare dispatch: per-file morphosyntax + DP alignment against gold.

use std::path::Path;
use std::sync::Arc;

use crate::api::{FileName, LanguageCode3};
use crate::params::CachePolicy;
use crate::pipeline::PipelineServices;
use crate::scheduling::FailureCategory;
use tracing::{info, warn};

use crate::store::{JobStore, RunnerJobSnapshot, unix_now};

use super::super::util::{FileRunTracker, apply_result_filename, classify_server_error};
use super::infer_batched::apply_merge_abbrev;

/// Dispatch compare: per-file morphosyntax + DP alignment against gold.
///
/// Each file is processed individually because it needs its own gold companion.
/// The compare orchestrator runs morphosyntax first, then DP-aligns against the
/// gold transcript and produces `%xsrep` tiers + a `.compare.csv` metrics file.
#[allow(clippy::too_many_arguments)]
pub(in crate::runner) async fn dispatch_compare(
    job: &RunnerJobSnapshot,
    store: &Arc<JobStore>,
    services: PipelineServices<'_>,
    file_texts: &[(String, String)],
    cache_policy: CachePolicy,
    mwt: &batchalign_chat_ops::morphosyntax::MwtDict,
    should_merge_abbrev: bool,
) {
    let job_id = &job.identity.job_id;
    let correlation_id = job.identity.correlation_id.as_str();
    let file_list = &job.pending_files;
    let lang: &LanguageCode3 = &job.dispatch.lang;

    for (filename, chat_text) in file_texts {
        let lifecycle = FileRunTracker::new(store, job_id, filename);
        // Skip gold files — they're companions, not inputs
        if crate::compare::is_gold_file(filename) {
            let now = unix_now();
            lifecycle.complete_without_result(now).await;
            continue;
        }

        // Find the gold file companion
        let gold_filename = crate::compare::gold_path_for(filename);

        // Read gold file: check if it's in the same batch, or read from disk
        let gold_text = file_texts
            .iter()
            .find(|(fn_, _)| *fn_ == gold_filename)
            .map(|(_, text)| text.clone());

        let gold_text = match gold_text {
            Some(text) => text,
            None => {
                // Try to read from the filesystem
                let file_index = file_list
                    .iter()
                    .find(|file| file.filename.as_ref() == filename.as_str())
                    .map(|file| file.file_index)
                    .unwrap_or(0);
                let gold_read_path = if job.filesystem.paths_mode {
                    if file_index < job.filesystem.source_paths.len() {
                        crate::compare::gold_path_for(&job.filesystem.source_paths[file_index])
                    } else {
                        crate::compare::gold_path_for(filename)
                    }
                } else {
                    format!("{}/input/{}", job.filesystem.staging_dir, gold_filename)
                };
                match tokio::fs::read_to_string(&gold_read_path).await {
                    Ok(text) => text,
                    Err(_) => {
                        lifecycle
                            .fail(
                                &format!(
                                    "No gold .cha file found for comparison. \
                                     main: {filename}, expected: {gold_filename}"
                                ),
                                FailureCategory::InputMissing,
                                unix_now(),
                            )
                            .await;
                        continue;
                    }
                }
            }
        };

        info!(job_id = %job_id, filename = %filename, "Running compare pipeline");

        match crate::compare::process_compare(
            chat_text,
            &gold_text,
            lang,
            services,
            cache_policy,
            mwt,
        )
        .await
        {
            Ok((mut chat_output, csv_output)) => {
                if should_merge_abbrev {
                    chat_output = apply_merge_abbrev(&chat_output);
                }

                let finished_at = unix_now();
                let file_index = file_list
                    .iter()
                    .find(|file| file.filename.as_ref() == filename.as_str())
                    .map(|file| file.file_index)
                    .unwrap_or(0);

                let write_path = if job.filesystem.paths_mode
                    && file_index < job.filesystem.output_paths.len()
                {
                    apply_result_filename(&job.filesystem.output_paths[file_index], filename)
                } else {
                    format!("{}/output/{}", job.filesystem.staging_dir, filename)
                };

                if let Some(parent) = Path::new(&write_path).parent() {
                    let _ = tokio::fs::create_dir_all(parent).await;
                }
                if let Err(e) = tokio::fs::write(&write_path, &chat_output).await {
                    warn!(error = %e, "Failed to write compare output");
                }

                // Write CSV metrics alongside the CHAT output
                let csv_path = write_path.replace(".cha", ".compare.csv");
                if let Err(e) = tokio::fs::write(&csv_path, &csv_output).await {
                    warn!(error = %e, "Failed to write compare CSV");
                }

                lifecycle
                    .complete_with_result(FileName::from(filename.as_str()), "chat", finished_at)
                    .await;
            }
            Err(e) => {
                lifecycle
                    .fail(&e.to_string(), classify_server_error(&e), unix_now())
                    .await;
            }
        }
    }

    let _ = correlation_id;
}

#[cfg(test)]
mod tests {
    use std::collections::{BTreeMap, HashMap};
    use std::sync::Arc;

    use tokio::sync::broadcast;
    use tokio_util::sync::CancellationToken;

    use super::*;
    use crate::api::{EngineVersion, FileStatusKind, JobId, JobStatus, NumSpeakers, UnixTimestamp};
    use crate::cache::UtteranceCache;
    use crate::db::JobDB;
    use crate::options::{CommandOptions, CommonOptions, CompareOptions};
    use crate::scheduling::{AttemptOutcome, WorkUnitKind};
    use crate::store::{
        FileStatus, Job, JobDispatchConfig, JobExecutionState, JobFilesystemConfig, JobIdentity,
        JobLeaseState, JobRuntimeControl, JobScheduleState, JobSourceContext,
    };
    use crate::worker::pool::WorkerPool;
    use crate::ws::BROADCAST_CAPACITY;

    /// Build a compare job containing a gold companion file.
    fn make_compare_job(job_id: &str, filename: &str, staging_dir: &str) -> Job {
        let mut file_statuses = HashMap::new();
        file_statuses.insert(
            filename.to_string(),
            FileStatus::new(FileName::from(filename)),
        );

        Job {
            identity: JobIdentity {
                job_id: JobId::from(job_id),
                correlation_id: format!("test-{job_id}"),
            },
            dispatch: JobDispatchConfig {
                command: "compare".into(),
                lang: "eng".into(),
                num_speakers: NumSpeakers(1),
                options: CommandOptions::Compare(CompareOptions {
                    common: CommonOptions::default(),
                    merge_abbrev: false.into(),
                }),
                runtime_state: BTreeMap::new(),
                debug_traces: false,
            },
            source: JobSourceContext {
                submitted_by: "127.0.0.1".into(),
                submitted_by_name: "localhost".into(),
                source_dir: String::new(),
            },
            filesystem: JobFilesystemConfig {
                filenames: vec![FileName::from(filename)],
                has_chat: vec![true],
                staging_dir: staging_dir.to_string(),
                paths_mode: false,
                source_paths: Vec::new(),
                output_paths: Vec::new(),
                before_paths: Vec::new(),
                media_mapping: String::new(),
                media_subdir: String::new(),
            },
            execution: JobExecutionState {
                status: JobStatus::Queued,
                file_statuses,
                results: Vec::new(),
                error: None,
                completed_files: 0,
            },
            schedule: JobScheduleState {
                submitted_at: UnixTimestamp(1.0),
                completed_at: None,
                next_eligible_at: None,
                num_workers: None,
                lease: JobLeaseState {
                    leased_by_node: None,
                    expires_at: None,
                    heartbeat_at: None,
                },
            },
            runtime: JobRuntimeControl {
                cancel_token: CancellationToken::new(),
                runner_active: false,
            },
        }
    }

    /// Gold companion files are skipped by compare, but their pre-opened batch
    /// infer attempts still need to be closed as successful.
    #[tokio::test]
    async fn gold_file_skip_finishes_open_attempt() {
        let tempdir = tempfile::tempdir().expect("tempdir");
        let db = Arc::new(JobDB::open(Some(tempdir.path())).await.expect("open db"));
        let (tx, _rx) = broadcast::channel(BROADCAST_CAPACITY);
        let store = Arc::new(JobStore::new(
            crate::config::ServerConfig::default(),
            Some(db.clone()),
            tx,
        ));
        let filename = "sample.gold.cha";
        store
            .submit(make_compare_job(
                "job-compare",
                filename,
                &tempdir.path().display().to_string(),
            ))
            .await
            .expect("submit job");

        let started_at = unix_now();
        store
            .mark_file_processing(&JobId::from("job-compare"), filename, started_at)
            .await;
        store
            .start_file_attempt(
                &JobId::from("job-compare"),
                filename,
                WorkUnitKind::BatchInfer,
                started_at,
            )
            .await;

        let snapshot = store
            .runner_snapshot(&JobId::from("job-compare"))
            .await
            .expect("runner snapshot");
        let pool = WorkerPool::new(crate::worker::pool::PoolConfig::default());
        let cache = UtteranceCache::sqlite(Some(tempdir.path().join("cache")))
            .await
            .expect("open cache");
        let engine_version = EngineVersion::from("compare-test");
        let services = PipelineServices {
            pool: &pool,
            cache: &cache,
            engine_version: &engine_version,
        };

        dispatch_compare(
            &snapshot,
            &store,
            services,
            &[(filename.to_string(), String::from("*PAR:\tgold"))],
            CachePolicy::UseCache,
            &batchalign_chat_ops::morphosyntax::MwtDict::default(),
            false,
        )
        .await;

        let attempts = db
            .load_attempts_for_job("job-compare")
            .await
            .expect("load attempts");
        assert_eq!(attempts.len(), 1);
        assert_eq!(attempts[0].work_unit_kind, WorkUnitKind::BatchInfer);
        assert_eq!(attempts[0].outcome, AttemptOutcome::Succeeded);

        let detail = store
            .get_job_detail(&JobId::from("job-compare"))
            .await
            .expect("job detail");
        assert_eq!(detail.file_statuses.len(), 1);
        assert_eq!(detail.file_statuses[0].status, FileStatusKind::Done);
    }
}
