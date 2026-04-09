# Managed Agents

Claude Managed Agents configs for kalshi-trading-bot. Each agent is defined by a YAML
config plus a system prompt that references a shared safety header.

## Layout

```
agents/
  _common/system_prompt_header.md   # safety header (read-only inputs, tarball final turn)
  kalshi-reporter.yaml              # agent config
  kalshi-reporter.system.md         # role-specific system prompt
  .deployed.json                    # [gitignored] agent ids returned by the API
  .usage.jsonl                      # [gitignored] per-session token/cost telemetry
  .last_run.json                    # [gitignored] heartbeat written by schedule.py
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
