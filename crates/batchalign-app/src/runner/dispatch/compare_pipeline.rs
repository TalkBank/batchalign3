//! Compare dispatch: per-file morphosyntax + DP alignment against gold.

use std::collections::HashMap;

use crate::api::{LanguageCode3, ReleasedCommand};
use crate::params::CachePolicy;
use crate::pipeline::PipelineServices;
use crate::recipe_runner::runtime::{
    output_write_path, plan_work_units_for_job, primary_output_artifact, sidecar_output_artifacts,
    write_text_output_artifact,
};
use crate::recipe_runner::work_unit::{CompareWorkUnit, PlannedWorkUnit};
use crate::runner::DispatchHostContext;
use crate::scheduling::FailureCategory;
use crate::text_batch::TextBatchFileInput;
use tracing::{info, warn};

use crate::store::{RunnerJobSnapshot, unix_now};

use super::super::util::{FileRunTracker, classify_server_error};
use super::infer_batched::apply_merge_abbrev;

/// Dispatch compare: per-file morphosyntax + DP alignment against gold.
///
/// Each file is processed individually because it needs its own gold companion.
/// The compare orchestrator runs morphosyntax first, then DP-aligns against the
/// gold transcript and produces `%xsrep`/`%xsmor` tiers + a `.compare.csv`
/// metrics file.
#[allow(clippy::too_many_arguments)]
pub(in crate::runner) async fn dispatch_compare(
    job: &RunnerJobSnapshot,
    host: &DispatchHostContext,
    services: PipelineServices<'_>,
    file_texts: &[TextBatchFileInput],
    cache_policy: CachePolicy,
    mwt: &batchalign_chat_ops::morphosyntax::MwtDict,
    should_merge_abbrev: bool,
) {
    let job_id = &job.identity.job_id;
    let correlation_id = &*job.identity.correlation_id;
    let file_list = &job.pending_files;
    let sink = host.sink().clone();
    let fallback_lang = crate::api::LanguageCode3::eng();
    let lang: &LanguageCode3 = job.dispatch.lang.as_resolved().unwrap_or(&fallback_lang);
    let compare_units: HashMap<String, CompareWorkUnit> =
        match plan_work_units_for_job(ReleasedCommand::Compare, job) {
            Ok(units) => units
                .into_iter()
                .filter_map(|unit| match unit {
                    PlannedWorkUnit::Compare(pair) => {
                        Some((pair.main.display_path.to_string(), pair))
                    }
                    _ => None,
                })
                .collect(),
            Err(error) => {
                for file in file_texts {
                    let filename = file.filename.as_ref();
                    if crate::compare::is_gold_file(filename) {
                        continue;
                    }
                    FileRunTracker::new(sink.as_ref(), job_id, filename)
                        .fail(
                            &format!("Compare planning failed: {error}"),
                            FailureCategory::Validation,
                            unix_now(),
                        )
                        .await;
                }
                return;
            }
        };

    for file in file_texts {
        let filename = file.filename.as_ref();
        let chat_text = file.chat_text.as_ref();
        let lifecycle = FileRunTracker::new(sink.as_ref(), job_id, filename);
        // Skip gold files — they're companions, not inputs
        if crate::compare::is_gold_file(filename) {
            let now = unix_now();
            lifecycle.complete_without_result(now).await;
            continue;
        }

        let Some(pair) = compare_units.get(filename) else {
            lifecycle
                .fail(
                    &format!("Compare planning produced no pair for {filename}"),
                    FailureCategory::Validation,
                    unix_now(),
                )
                .await;
            continue;
        };
        let gold_filename = pair.gold.display_path.as_ref();

        // Read gold file: check if it's in the same batch, or read from disk
        let gold_text = file_texts
            .iter()
            .find(|candidate| candidate.filename.as_ref() == gold_filename)
            .map(|candidate| candidate.chat_text.to_string());

        let gold_text = match gold_text {
            Some(text) => text,
            None => match tokio::fs::read_to_string(&pair.gold.source_path).await {
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
            },
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
            Ok(outputs) => {
                let mut chat_output = outputs.chat_output;
                let csv_output = outputs.metrics_csv;
                if should_merge_abbrev {
                    chat_output = apply_merge_abbrev(&chat_output);
                }

                let finished_at = unix_now();
                let file_index = file_list
                    .iter()
                    .find(|file| file.filename.as_ref() == filename)
                    .map(|file| file.file_index)
                    .unwrap_or(0);
                let primary_output =
                    primary_output_artifact(ReleasedCommand::Compare, &pair.main.display_path);
                let csv_output_artifact =
                    sidecar_output_artifacts(ReleasedCommand::Compare, &pair.main.display_path)
                        .into_iter()
                        .find(|artifact| artifact.display_path.as_ref().ends_with(".compare.csv"))
                        .expect("compare command must emit a compare.csv sidecar");

                if let Err(e) = write_text_output_artifact(
                    &job.filesystem,
                    file_index,
                    &primary_output.display_path,
                    &chat_output,
                )
                .await
                {
                    warn!(error = %e, "Failed to write compare output");
                }

                // Write CSV metrics alongside the CHAT output
                let csv_path = output_write_path(
                    &job.filesystem,
                    file_index,
                    &csv_output_artifact.display_path,
                );
                if let Err(e) = tokio::fs::write(&csv_path, &csv_output).await {
                    warn!(error = %e, "Failed to write compare CSV");
                }

                lifecycle
                    .complete_with_result(
                        primary_output.display_path.clone(),
                        primary_output.content_type,
                        finished_at,
                    )
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
    use crate::api::{
        DisplayPath, EngineVersion, FileStatusKind, JobId, JobStatus, NumSpeakers, ReleasedCommand,
        UnixTimestamp,
    };
    use crate::api::{LanguageCode3, LanguageSpec};
    use crate::cache::UtteranceCache;
    use crate::db::JobDB;
    use crate::options::{CommandOptions, CommonOptions, CompareOptions};
    use crate::runner::DispatchHostContext;
    use crate::scheduling::{AttemptOutcome, WorkUnitKind};
    use crate::store::{
        FileStatus, Job, JobDispatchConfig, JobExecutionState, JobFilesystemConfig, JobIdentity,
        JobLeaseState, JobRuntimeControl, JobScheduleState, JobSourceContext, JobStore,
    };
    use crate::worker::pool::WorkerPool;
    use crate::ws::BROADCAST_CAPACITY;

    /// Build a compare job containing a gold companion file.
    fn make_compare_job(job_id: &str, filename: &str, staging_dir: &str) -> Job {
        let mut file_statuses = HashMap::new();
        file_statuses.insert(
            filename.to_string(),
            FileStatus::new(DisplayPath::from(filename)),
        );

        Job {
            identity: JobIdentity {
                job_id: JobId::from(job_id),
                correlation_id: format!("test-{job_id}").into(),
            },
            dispatch: JobDispatchConfig {
                command: ReleasedCommand::Compare,
                lang: LanguageSpec::Resolved(LanguageCode3::eng()),
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
                source_dir: Default::default(),
            },
            filesystem: JobFilesystemConfig {
                filenames: vec![DisplayPath::from(filename)],
                has_chat: vec![true],
                staging_dir: batchalign_types::paths::ServerPath::new(staging_dir),
                paths_mode: false,
                source_paths: Vec::new(),
                output_paths: Vec::new(),
                before_paths: Vec::new(),
                media_mapping: Default::default(),
                media_subdir: Default::default(),
                source_dir: Default::default(),
            },
            execution: JobExecutionState {
                status: JobStatus::Queued,
                file_statuses,
                results: Vec::new(),
                error: None,
                completed_files: 0,
                batch_progress: None,
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
        let services = PipelineServices::new(&pool, &cache, &engine_version);
        let host = DispatchHostContext::from_store(store.clone());

        dispatch_compare(
            &snapshot,
            &host,
            services,
            &[TextBatchFileInput::new(
                filename.to_string(),
                String::from("*PAR:\tgold"),
            )],
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
