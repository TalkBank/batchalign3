/** Cache backend status indicator.
 *
 * Displays the active utterance cache backend (`sqlite` or `redis`)
 * and, when Redis is configured, whether the connection is alive.
 * Designed for the dashboard right column alongside MemoryPanel.
 *
 * Data source: `cache_backend` and `redis_cache_connected` fields
 * from the `/health` endpoint.
 */

import type { HealthResponse } from "../types";

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

interface CacheStatusPanelProps {
  health: HealthResponse | undefined;
}

/** Compact cache backend status badge row.
 *
 * Renders only when the health response includes cache information.
 * Shows the backend name and, for Redis, a connected/disconnected dot.
 */
export function CacheStatusPanel({ health }: CacheStatusPanelProps) {
  if (!health) return null;

  const backend = health.cache_backend;
  if (!backend) return null;

  const redisEnabled = health.redis_cache_enabled ?? false;
  const redisConnected = health.redis_cache_connected ?? false;

  return (
    <div className="bg-white border border-gray-200 rounded-lg overflow-hidden">
      <div className="flex items-center justify-between px-4 py-2.5">
        <div className="flex items-center gap-2">
          <svg
            className="w-4 h-4 text-gray-400"
            fill="none"
            viewBox="0 0 24 24"
            strokeWidth={1.5}
            stroke="currentColor"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              d="M20.25 6.375c0 2.278-3.694 4.125-8.25 4.125S3.75 8.653 3.75 6.375m16.5 0c0-2.278-3.694-4.125-8.25-4.125S3.75 4.097 3.75 6.375m16.5 0v11.25c0 2.278-3.694 4.125-8.25 4.125s-8.25-1.847-8.25-4.125V6.375m16.5 0v3.75c0 2.278-3.694 4.125-8.25 4.125s-8.25-1.847-8.25-4.125v-3.75"
            />
          </svg>
          <span className="text-sm font-semibold text-gray-700">Cache</span>
        </div>

        <div className="flex items-center gap-2">
          {/* Backend label */}
          <span className="text-xs font-mono text-gray-500 bg-gray-100 px-2 py-0.5 rounded">
            {backend}
          </span>

          {/* Redis connection status (only shown when Redis is configured) */}
          {redisEnabled && (
            <span
              className={`inline-flex items-center gap-1 px-2 py-0.5 rounded text-[11px] font-medium ${
                redisConnected
                  ? "text-emerald-700 bg-emerald-50"
                  : "text-red-700 bg-red-50"
              }`}
            >
              <span
                className={`inline-block w-1.5 h-1.5 rounded-full ${
                  redisConnected ? "bg-emerald-500" : "bg-red-500"
                }`}
              />
              {redisConnected ? "connected" : "disconnected"}
            </span>
          )}
        </div>
      </div>
    </div>
  );
}
