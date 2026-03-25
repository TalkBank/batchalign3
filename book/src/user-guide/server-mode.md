# Server Mode

**Last modified:** 2026-03-24 21:21 EDT

Batchalign includes a built-in HTTP server managed by `batchalign3 serve ...`.
The CLI is always a client: it either talks to an explicit remote server
(`--server`) or to a local daemon.

## Current routing rules

- With `--server URL`, the CLI submits supported jobs to that server.
- Without `--server`, the CLI tries the local auto-daemon when `auto_daemon`
  is enabled in `~/.batchalign3/server.yaml`.
- `transcribe` and `avqi` currently ignore explicit remote `--server` because
  the remote server cannot access client-local audio files.
- When the main local daemon lacks a needed capability, the CLI may try a
  sidecar daemon for local `transcribe`, `benchmark`, or `avqi` work.

## Start a server

Foreground:

```bash
batchalign3 serve start --foreground
```

Background:

```bash
batchalign3 serve start
```

Useful flags:

```bash
batchalign3 serve start --foreground --port 8000
batchalign3 serve start --foreground --config ~/server.yaml
batchalign3 serve start --foreground --warmup minimal
batchalign3 serve start --foreground --test-echo
```

## Check and stop a server

```bash
batchalign3 serve status
batchalign3 serve status --server http://myserver:8000
batchalign3 serve stop
```

Inspect remote jobs:

```bash
batchalign3 jobs --server http://myserver:8000
batchalign3 jobs --server http://myserver:8000 <JOB_ID>
```

## Server configuration

Default config path:

```text
~/.batchalign3/server.yaml
```

Minimal example:

```yaml
default_lang: eng
port: 8000
max_concurrent_jobs: 8
auto_daemon: true
warmup_commands: [morphotag, align, transcribe]
media_roots: []
media_mappings: {}
```

Important keys:

- `port` — server listen port
- `host` — bind address (defaults to `0.0.0.0`)
- `max_concurrent_jobs` — `0` means auto-tune
- `auto_daemon` — allow the CLI to auto-start a local daemon when no `--server` is given
- `warmup_commands` — list of commands to pre-warm (see [Worker Tuning](worker-tuning.md))
- `media_roots` — directories searched for media
- `media_mappings` — named client-path to server-path mappings
- `memory_tier` — override auto-detected tier: `small`, `medium`, `large`, `fleet`
- `memory_gate_mb` — host headroom reserve (default: tier-dependent, 2000-8000 MB)
- `gpu_startup_mb` / `stanza_startup_mb` / `io_startup_mb` — per-profile startup reservation overrides (0 = tier default)
- `worker_idle_timeout_s` — shut down idle workers after this many seconds (default: tier-dependent — 60s Small, 300s Medium, 600s Large/Fleet)
- `worker_health_interval_s` — health check frequency in seconds (default: 30)
- `job_ttl_days` — auto-delete completed jobs after this many days (default: 7)

OTLP tracing can be enabled by setting `BATCHALIGN_OTLP_ENDPOINT`
(or `OTEL_EXPORTER_OTLP_ENDPOINT`) in the server environment.

Reference example files live in `examples/server.yaml` and
`examples/launchd.plist`.

`server.yaml` uses a strict schema. Unknown keys are rejected at startup
instead of being silently ignored, so stale config like `warmup: false` must
be updated to the current `warmup_commands: []` form.

## Remote use

Commands that support explicit remote dispatch look like this:

```bash
batchalign3 --server http://myserver:8000 morphotag corpus/ -o output/
batchalign3 --server http://myserver:8000 align corpus/ -o output/
```

Health checks:

```bash
curl -s http://myserver:8000/health | python3 -m json.tool
batchalign3 serve status --server http://myserver:8000
```

The `/health` response includes a `capabilities` list of all commands the server
can handle. If a command is missing, the server's Python environment is likely
missing a required package. See
[Troubleshooting: "Command not supported"](troubleshooting.md#command-not-supported-or-missing-commands).

## launchd example (macOS)

For always-on macOS hosts, use `examples/launchd.plist` as a template and
update the binary path, username, and log paths before installing it as a
LaunchDaemon.
