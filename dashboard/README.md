# Harnyx Miner Dashboard

A self-hosted operator dashboard for Harnyx (SN67) miners. It tracks two things:

1. **Your local iteration history** -- every `local-eval-report-*.json` /
   `local-benchmark-report-*.json` produced by `harnyx-miner-local-eval` /
   `harnyx-miner-local-benchmark` in a directory you point it at, trended over
   time (score, cost, win/loss vs. champion).
2. **Public SN67 network state** -- current champion and recent completed
   batches, via the same public, unauthenticated `/v1/monitoring/...`
   endpoints the `harnyx-miner` CLI already uses
   (`harnyx_miner.platform_monitoring.PlatformMonitoringClient`).

It does **not** need a wallet, hotkey, or any provider API key to run --
only `PLATFORM_BASE_URL` (loaded the same way the rest of the miner tooling
loads it, from `.env` at the repo root).

A specific hotkey's standing (its own score/rank across batches) is not yet
resolvable from public data alone -- see the "Hotkey standing" card in the
UI for why, and what would be needed to add it.

## Run it

From the repo root:

```bash
uv run --package harnyx-dashboard harnyx-dashboard --reports-dir . --port 8787
```

Then open `http://127.0.0.1:8787`. Generate more data points by running
`harnyx-miner-local-eval` / `harnyx-miner-local-benchmark` in the same
`--reports-dir` -- the dashboard picks up new reports on its next poll.

## CLI options

| Flag | Purpose |
|------|---------|
| `--reports-dir` | Directory to scan for report JSON files (default: cwd) |
| `--host` / `--port` | Bind address (default: `127.0.0.1:8787`) |
| `--cache-seconds` | TTL for cached public-API responses (default: `60`) |
| `--uid` / `--hotkey` | Surfaced on the "Hotkey standing" card for your own reference |

All of the above also read from `DASHBOARD_REPORTS_DIR`, `DASHBOARD_HOST`,
`DASHBOARD_PORT`, `DASHBOARD_CACHE_SECONDS`, `DASHBOARD_UID`,
`DASHBOARD_HOTKEY` env vars if the flag is omitted.
