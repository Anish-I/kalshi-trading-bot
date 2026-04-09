# Agents

Two provider backends for kalshi-trading-bot analysis agents:

**`claude-headless` (default)** — uses `claude -p` subprocess. Authenticates via
the existing Claude Code OAuth (Claude Max subscription). Zero API key needed,
zero per-session cost. Ideal for batch tasks that run on the same machine as
the data (daily reports, post-mortems, local research). Used by the daily
PnL reporter MVP.

**`managed-agents`** — uses the Anthropic Managed Agents beta API
(`managed-agents-2026-04-01`). Runs in cloud-provisioned containers. Requires
`ANTHROPIC_API_KEY` (Max subscription does NOT cover this meter). Reserved
for future use cases that genuinely need cloud isolation — e.g. P5 parameter
search with cloud GPUs, or agents that must not have access to the local
filesystem.

Run a task with:
```
python scripts/agents/schedule.py reporter                       # default: claude-headless
python scripts/agents/schedule.py reporter --provider claude-headless
python scripts/agents/schedule.py reporter --provider managed-agents   # needs API key
```

Each agent is defined by a YAML config plus a system prompt that references a
shared safety header. The system prompts come in two flavors: `*.system.md`
(managed-agents version with /work/inputs/ + base64 tarball convention) and
`*.headless.system.md` (path-agnostic, direct filesystem reads).

## Layout

```
agents/
  _common/system_prompt_header.md          # safety header (read-only inputs)
  kalshi-reporter.yaml                     # agent config (shared by both providers)
  kalshi-reporter.system.md                # managed-agents variant (tarball round-trip)
  kalshi-reporter.headless.system.md       # claude-headless variant (direct filesystem)
  .deployed.json                           # [gitignored] managed-agents IDs
  .usage.jsonl                             # [gitignored] per-session telemetry
  .last_run.json                           # [gitignored] heartbeat from schedule.py
```

## Versioning

Bump `version:` in the YAML, edit the system prompt if needed, commit. On the next
`deploy_agents.py` run, the script will call `beta.agents.update()` for existing
entries in `.deployed.json`. Never edit `.deployed.json` by hand.

## Redeploy

```
python scripts/agents/deploy_agents.py               # apply YAML changes
python scripts/agents/deploy_agents.py --dry-run     # preview payload, no API call
```

`ANTHROPIC_API_KEY` must be exported (the scripts exit 2 otherwise).

## Rollback

Managed Agents are the scheduled side. To disable the daily reporter without
touching code:

```
schtasks /Change /TN "KalshiReporter" /DISABLE
```

(The Task Scheduler entry is **not** created by this repo — the user registers it
manually once the API key is in place.)

## Auditing a snapshot

Before trusting a new agent type, inspect the tarball that will be sent:

```
python scripts/agents/snapshot.py --task reporter --out tmp_snapshot.tar.gz
tar tzf tmp_snapshot.tar.gz
```

Only `order_ledger` (7-day window), `trade_journal`, the 5 trader state JSONs, a
fresh `family_scorecard.parquet`, and `manifest.json` should appear. No PEM
files, no `.env`, no credentials.
