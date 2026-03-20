/** Presentational job-detail view.
 *
 * This component owns route-local view logic such as file filters, pagination,
 * and section layout. It deliberately does not fetch data or discover which
 * server owns the job; those concerns live in `useJobPageController`.
 */
import { useState } from "react";
import { Layout } from "./Layout";
import { ProgressBar } from "./ProgressBar";
import { ActionButtons } from "./ActionButtons";
import { StatusSummaryStrip } from "./StatusSummaryStrip";
import { ErrorPanel } from "./ErrorPanel";
import { FilterTabs } from "./FilterTabs";
import { PaginatedFileList } from "./PaginatedFileList";
import { useFileFilters } from "../hooks/useFileFilters";
import type { JobInfo, JobListItem } from "../types";

/** Tiny copy-to-clipboard button that shows a brief "Copied" tooltip. */
function CopyButton({ text, label }: { text: string; label: string }) {
  const [copied, setCopied] = useState(false);
  async function handleCopy() {
    try {
      await navigator.clipboard.writeText(text);
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    } catch {
      // Clipboard API may be blocked in some contexts
    }
  }
  return (
    <button
      type="button"
      onClick={handleCopy}
      className="inline-flex items-center text-zinc-300 hover:text-zinc-500 transition-colors ml-1.5"
      aria-label={`Copy ${label}`}
      title={copied ? "Copied!" : `Copy ${label}`}
    >
      {copied ? (
        <svg className="w-3.5 h-3.5 text-emerald-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
        </svg>
      ) : (
        <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
            d="M8 16H6a2 2 0 01-2-2V6a2 2 0 012-2h8a2 2 0 012 2v2m-6 12h8a2 2 0 002-2v-8a2 2 0 00-2-2h-8a2 2 0 00-2 2v8a2 2 0 002 2z" />
        </svg>
      )}
    </button>
  );
}
import {
  commandStyle,
  formatJsonDisplay,
  formatDuration,
  progressPercent,
  relativeTime,
  shortPath,
  statusDotColor,
  submitterName,
  displayLang,
  isDefaultLang,
} from "../utils";

/** Inputs required to render a fully resolved job detail page. */
type JobDetailPageViewProps = {
  detail: JobInfo;
  wsJob: JobListItem | undefined;
  multiServer: boolean;
  effectiveServer: string;
  serverBase: string;
  onDeleted: () => void;
};

/** Render one job detail page from controller-supplied state plus live summary data. */
export function JobDetailPageView({
  detail,
  wsJob,
  multiServer,
  effectiveServer,
  serverBase,
  onDeleted,
}: JobDetailPageViewProps) {
  const fileStatuses = detail.file_statuses;
  const {
    activeTab,
    setActiveTab,
    searchQuery,
    setSearchQuery,
    page,
    setPage,
    counts,
    errorGroups,
    filteredFiles,
    pageFiles,
    totalPages,
    pageSize,
  } = useFileFilters(fileStatuses);

  const completedFiles = wsJob?.completed_files ?? detail.completed_files;
  const currentStatus = wsJob?.status ?? detail.status;
  const isActive = currentStatus === "queued" || currentStatus === "running";
  const isRunning = currentStatus === "running";
  const pct = progressPercent(completedFiles, detail.total_files);
  const [cmdBg, cmdText] = commandStyle(detail.command);
  const host = submitterName(detail.submitted_by_name, detail.submitted_by);
  const commandArgsJson = formatJsonDisplay(detail.options);

  return (
    <Layout>
      {/* Navigation stays in the view so route shells do not accumulate markup. */}
      <div className="mb-5">
        <a
          href="/dashboard"
          className="text-xs text-zinc-400 hover:text-zinc-600 transition-colors no-underline"
        >
          &larr; Back to jobs
        </a>
      </div>

      <div className="bg-white rounded-lg border border-zinc-200">
        {/* Header and job-scoped action controls. */}
        <div className="px-5 pt-5 pb-4 border-b border-zinc-100">
          <div className="flex items-start justify-between gap-4">
            <div className="min-w-0">
              <div className="flex items-center gap-3 mb-2">
                <span
                  className={`inline-block px-2.5 py-1 rounded text-xs font-mono font-semibold uppercase tracking-wider ${cmdBg} ${cmdText}`}
                >
                  {detail.command}
                </span>
                <span className="inline-flex items-center gap-1.5">
                  <span
                    className={`inline-block w-2 h-2 rounded-full ${statusDotColor(currentStatus)} ${
                      isRunning ? "status-dot-pulse" : ""
                    }`}
                  />
                  <span className="text-sm text-zinc-500 capitalize">
                    {currentStatus}
                  </span>
                </span>
                {multiServer && effectiveServer && (
                  <span className="text-xs px-1.5 py-0.5 rounded bg-zinc-100 text-zinc-400">
                    {effectiveServer}
                  </span>
                )}
              </div>

              <span className="font-mono text-xs text-zinc-400">
                {detail.job_id}
                <CopyButton text={detail.job_id} label="job ID" />
              </span>
            </div>

            <ActionButtons
              jobId={detail.job_id}
              status={currentStatus}
              serverBase={serverBase}
              onDeleted={onDeleted}
            />
          </div>
        </div>

        {/* Static metadata and summary counts. */}
        <div className="px-5 py-4 border-b border-zinc-100">
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-y-3 gap-x-6 text-sm">
            <div>
              <div className="text-[11px] text-zinc-400 uppercase tracking-wider mb-0.5">
                Files
              </div>
              <div className="font-mono text-zinc-700">
                {completedFiles}
                <span className="text-zinc-300"> / </span>
                {detail.total_files}
                {isActive && (
                  <span className="text-zinc-400 text-xs ml-1">({pct}%)</span>
                )}
              </div>
            </div>
            <div>
              <div className="text-[11px] text-zinc-400 uppercase tracking-wider mb-0.5">
                Submitted
              </div>
              <div className="text-zinc-700">{relativeTime(detail.submitted_at)}</div>
            </div>
            <div>
              <div className="text-[11px] text-zinc-400 uppercase tracking-wider mb-0.5">
                Duration
              </div>
              <div className="font-mono text-zinc-700">
                {formatDuration(detail.duration_s) || "\u2014"}
              </div>
            </div>
            <div>
              <div className="text-[11px] text-zinc-400 uppercase tracking-wider mb-0.5">
                Workers
              </div>
              <div className="font-mono text-zinc-700">{detail.num_workers ?? "\u2014"}</div>
            </div>
          </div>

          {detail.source_dir && (
            <div
              className="mt-3 text-xs text-zinc-500 font-mono truncate"
              title={detail.source_dir}
            >
              {shortPath(detail.source_dir)}
            </div>
          )}

          {(host || (detail.lang && !isDefaultLang(detail.lang))) && (
            <div className="flex items-center gap-3 mt-3 text-xs text-zinc-400">
              {host && (
                <span>
                  <span className="text-zinc-500">from</span>{" "}
                  <span className="font-mono">{host}</span>
                </span>
              )}
              {detail.lang && !isDefaultLang(detail.lang) && (
                <span className="font-mono uppercase">{displayLang(detail.lang)}</span>
              )}
            </div>
          )}

          {/* Original submission args are critical for debugging reruns and
              understanding exactly which engine/flags produced the job. */}
          {commandArgsJson && (
            <div className="mt-4">
              <div className="flex items-center gap-1 text-[11px] text-zinc-400 uppercase tracking-wider mb-1.5">
                Command Args
                <CopyButton text={commandArgsJson} label="command args" />
              </div>
              <pre className="overflow-x-auto rounded-md border border-zinc-200 bg-zinc-50 px-3 py-2 text-[11px] leading-5 text-zinc-700">
                {commandArgsJson}
              </pre>
            </div>
          )}
        </div>

        {/* Progress is only meaningful for active jobs. */}
        {isActive && (
          <div className="px-5 py-3 border-b border-zinc-100">
            <ProgressBar
              completed={completedFiles}
              total={detail.total_files}
              animated={isRunning}
            />
          </div>
        )}

        {/* Job-level failures render above file-level breakdowns. */}
        {detail.error && (
          <div className="mx-5 mt-4 bg-red-50 border border-red-100 rounded-lg p-3 text-sm text-red-700">
            {detail.error}
          </div>
        )}

        {/* File-level summary and filtering controls. */}
        {fileStatuses.length > 0 && (
          <div className="px-5 pt-4">
            <StatusSummaryStrip counts={counts} onStatusClick={setActiveTab} />
          </div>
        )}

        {errorGroups.length > 0 && (
          <div className="px-5 pt-3">
            <ErrorPanel errorGroups={errorGroups} />
          </div>
        )}

        {/* Paginated file table plus tab/search controls. */}
        <div className="px-5 pt-4 pb-5">
          <div className="mb-3">
            <FilterTabs
              activeTab={activeTab}
              counts={counts}
              searchQuery={searchQuery}
              onTabChange={setActiveTab}
              onSearchChange={setSearchQuery}
            />
          </div>

          <PaginatedFileList
            pageFiles={pageFiles}
            page={page}
            totalPages={totalPages}
            totalFiltered={filteredFiles.length}
            pageSize={pageSize}
            onPageChange={setPage}
          />
        </div>
      </div>
    </Layout>
  );
}
