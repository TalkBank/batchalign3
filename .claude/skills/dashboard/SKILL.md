---
name: dashboard
description: Debug the React dashboard (SPA, Tauri desktop, API contract). Use for blank screens, stale data, build failures, API drift, or WebSocket issues.
disable-model-invocation: true
allowed-tools: Bash, Read, Glob, Grep, Agent
---

# Dashboard Debugging

Diagnose problems with the batchalign3 React dashboard. `$ARGUMENTS` describes the symptom.

## Architecture

```
React SPA (frontend/) → axum REST API (batchalign-app) → WebSocket (real-time updates)
                       → Tauri desktop (apps/dashboard-desktop/) [optional wrapper]
```

The dashboard is a React SPA served by the Rust axum server at `/dashboard`. TypeScript types are generated from the Rust API schema.

## Step 1: Identify the Layer

| Symptom | Likely Layer | Check First |
|---------|-------------|-------------|
| Blank screen | Asset serving or build | Does `~/.batchalign3/dashboard/index.html` exist? |
| Stale data | WebSocket or React Query | Browser DevTools → Network → WS tab |
| Wrong types / TS errors | API drift | `scripts/check_dashboard_api_drift.sh` |
| Layout broken | React/Tailwind | Browser DevTools → Elements |
| API 404/500 | Rust server | `curl http://localhost:8000/api/health` |

## Step 2: Symptom-Based Triage

### "Blank screen / nothing renders"

```bash
# Check if dashboard is built
ls ~/.batchalign3/dashboard/index.html

# Check if server is running
curl -s http://localhost:8000/api/health

# Check if dashboard route returns HTML
curl -s http://localhost:8000/dashboard | head -5
```

- Check `BATCHALIGN_DASHBOARD_DIR` env var override
- Check browser DevTools console for JS errors
- Check network tab — are JS/CSS assets loading?

### "Data not updating / stale"

- Check WebSocket connection in browser DevTools → Network → WS
- Check React Query devtools (included in dev mode)
- Force invalidation: browser console `queryClient.invalidateQueries()`
- Check server is sending updates: `wscat -c ws://localhost:8000/ws`

### "API errors / wrong data shape"

```bash
# Check for API type drift
scripts/check_dashboard_api_drift.sh

# Regenerate TypeScript types from Rust
scripts/generate_dashboard_api_types.sh

# Compare what server returns vs what TS expects
curl -s http://localhost:8000/api/jobs | python3 -m json.tool | head -20
```

### "Build fails"

```bash
# Frontend build
cd frontend && npm run build 2>&1 | head -30

# Check if deps are installed
cd frontend && npm ci

# Check TypeScript errors
cd frontend && npx tsc --noEmit
```

### "Tauri desktop issues"

```bash
# Check Tauri build
cd apps/dashboard-desktop && npm run build 2>&1 | head -20

# Is Vite dev server running?
cd frontend && npm run dev
```

## Step 3: Development Workflow

### Hot reload development

```bash
# Terminal 1: Start the Rust server
cargo run -p batchalign-cli -- serve --port 8000

# Terminal 2: Start Vite dev server with proxy
cd frontend && npm run dev
# Opens at http://localhost:5173 with hot reload
```

### Regenerate API types after Rust changes

```bash
scripts/generate_dashboard_api_types.sh
scripts/check_dashboard_api_drift.sh    # CI gate
```

### Production build

```bash
cd frontend && npm run build
# Output: frontend/dist/
# Copy to: ~/.batchalign3/dashboard/
```

## Step 4: Testing

```bash
# Playwright smoke tests
scripts/run_react_dashboard_smoke.sh

# Real-server E2E (requires running server)
BATCHALIGN_REAL_SERVER_E2E=1 scripts/run_react_dashboard_smoke.sh
```

## Key Files

| Purpose | Path |
|---------|------|
| React app entry | `frontend/src/App.tsx` |
| API client / React Query hooks | `frontend/src/api/` |
| Generated TypeScript types | `frontend/src/types/` |
| Tailwind config | `frontend/tailwind.config.js` |
| Vite config | `frontend/vite.config.ts` |
| Playwright tests | `frontend/e2e/` |
| API type generation script | `scripts/generate_dashboard_api_types.sh` |
| API drift check script | `scripts/check_dashboard_api_drift.sh` |
| Smoke test runner | `scripts/run_react_dashboard_smoke.sh` |
| Rust server API routes | `crates/batchalign-app/src/` |
| Dashboard asset serving | `crates/batchalign-app/src/dashboard.rs` (if exists) |
| Tauri desktop wrapper | `apps/dashboard-desktop/` (if exists) |
