/** Visual server status indicator with manual start/stop controls.
 *
 * Shows a colored dot (green/yellow/red) with a label describing the server
 * state. In desktop mode, includes a restart/start button when the server
 * is stopped or has crashed.
 */

import type { ServerLifecycleState } from "../../hooks/useServerLifecycle";

interface ServerStatusBarProps {
  lifecycle: ServerLifecycleState;
}

/** Dot color class for each status. */
function dotColor(status: ServerLifecycleState["status"]): string {
  switch (status) {
    case "running":
      return "bg-emerald-500";
    case "starting":
      return "bg-amber-400 status-dot-pulse";
    case "stopped":
      return "bg-red-500";
    case "not-found":
      return "bg-red-500";
    case "external":
      return "bg-gray-400";
  }
}

/** Human-readable label for each status. */
function statusLabel(status: ServerLifecycleState["status"]): string {
  switch (status) {
    case "running":
      return "Server running";
    case "starting":
      return "Server starting...";
    case "stopped":
      return "Server stopped";
    case "not-found":
      return "batchalign3 not found";
    case "external":
      return "External server";
  }
}

export function ServerStatusBar({ lifecycle }: ServerStatusBarProps) {
  const { status, start, stop, error } = lifecycle;

  return (
    <div className="flex items-center justify-between bg-white border border-gray-200 rounded-lg px-4 py-2.5">
      <div className="flex items-center gap-2.5">
        <span
          className={`inline-block w-2.5 h-2.5 rounded-full ${dotColor(status)}`}
        />
        <span className="text-sm text-gray-700">{statusLabel(status)}</span>
        {lifecycle.health.health?.version && status === "running" && (
          <span className="text-xs text-gray-400">
            v{lifecycle.health.health.version}
          </span>
        )}
      </div>

      <div className="flex items-center gap-2">
        {error && (
          <span className="text-xs text-red-600 max-w-xs truncate">
            {error}
          </span>
        )}

        {status === "not-found" && (
          <span className="text-xs text-gray-500">
            Install with: uv tool install batchalign3
          </span>
        )}

        {status === "stopped" && (
          <button
            type="button"
            onClick={start}
            className="text-xs font-medium text-indigo-600 hover:text-indigo-800 transition-colors"
          >
            Start Server
          </button>
        )}

        {status === "running" && (
          <button
            type="button"
            onClick={stop}
            className="text-xs font-medium text-gray-400 hover:text-red-600 transition-colors"
          >
            Stop
          </button>
        )}
      </div>
    </div>
  );
}
