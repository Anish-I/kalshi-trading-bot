"""Agent orchestrator — runs strategist agents and writes proposals.

Flow per agent:
1. Load YAML from agents/<name>.yaml
2. Build snapshot via scripts.agents.snapshot (optional — mock-claude skips it)
3. Invoke claude -p via run_claude_headless.run_strategist with strategist prompt
4. Parse output JSON → extract proposed_config + rationale + git_patch
5. Call codex_review.review() to get a CodexVerdict
6. Write combined proposal to D:/kalshi-data/agent_proposals/pending/<id>.json
7. Write telemetry to agents/.usage.jsonl

CLI:
    python scripts/agents/orchestrator.py --agent weather-strategist
    python scripts/agents/orchestrator.py --agent weather-strategist --mock-codex
    python scripts/agents/orchestrator.py --agent weather-strategist --mock-claude --mock-codex
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

REPO_ROOT = Path(__file__).resolve().parents[2]
AGENTS_DIR = REPO_ROOT / "agents"

# Allow `python scripts/agents/orchestrator.py ...` by ensuring repo root is on sys.path
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Default proposal store. Can be overridden via env for tests.
DEFAULT_PROPOSAL_ROOT = Path(
    os.environ.get("KALSHI_AGENT_PROPOSAL_ROOT", "D:/kalshi-data/agent_proposals")
)

ORCHESTRATOR_TIMEOUT_S = 900

# Fallback mock proposal used in --mock-claude mode so tests/E2E can run offline.
MOCK_PROPOSAL_TEMPLATE = {
    "proposed_config": {
        "ENTRY_PRICE_MIN_CENTS": 5,
        "ENTRY_PRICE_MAX_CENTS": 95,
    },
    "current_config": {
        "ENTRY_PRICE_MIN_CENTS": 0,
        "ENTRY_PRICE_MAX_CENTS": 100,
    },
    "rationale": (
        "[MOCK CLAUDE] No real model was invoked. Synthetic proposal for "
        "infrastructure testing only."
    ),
    "success_criteria_snapshot": {
        "rolling_7d_win_rate": 0.38,
        "rolling_7d_ev_per_trade_cents": 12.0,
        "rolling_7d_max_drawdown_cents": 1800,
    },
    "git_patch": (
        "--- a/scripts/weather_trade.py\n"
        "+++ b/scripts/weather_trade.py\n"
        "@@ -1,0 +1,3 @@\n"
        "+# MOCK: entry price bounds proposed by strategist agent\n"
        "+ENTRY_PRICE_MIN_CENTS = 5\n"
        "+ENTRY_PRICE_MAX_CENTS = 95\n"
    ),
}


def _load_yaml(path: Path) -> dict:
    """Minimal YAML loader — prefers PyYAML, falls back to a tiny subset parser."""
    text = path.read_text(encoding="utf-8")
    try:
        import yaml  # type: ignore
        return yaml.safe_load(text) or {}
    except ImportError:
        return _tiny_yaml(text)


def _tiny_yaml(text: str) -> dict:
    """Extremely small YAML subset parser sufficient for our agent YAMLs.

    Supports: top-level key: value, nested lists of mappings via `-` at any indent,
    scalar ints/floats/bools/strings, and simple inline dicts `{k: v, ...}`.
    """
    import ast

    def parse_scalar(s: str) -> Any:
        s = s.strip()
        if not s:
            return None
        if s.startswith("{") and s.endswith("}"):
            # Convert YAML-ish inline mapping to Python dict literal.
            body = s[1:-1].strip()
            if not body:
                return {}
            parts: list[str] = []
            depth = 0
            cur = ""
            for ch in body:
                if ch == "," and depth == 0:
                    parts.append(cur)
                    cur = ""
                    continue
                if ch in "[{":
                    depth += 1
                elif ch in "]}":
                    depth -= 1
                cur += ch
            if cur:
                parts.append(cur)
            out: dict = {}
            for p in parts:
                if ":" not in p:
                    continue
                k, v = p.split(":", 1)
                out[k.strip()] = parse_scalar(v)
            return out
        if s.startswith("[") and s.endswith("]"):
            try:
                return ast.literal_eval(s)
            except Exception:
                return s
        if s.lower() in ("true", "false"):
            return s.lower() == "true"
        if s.lower() in ("null", "none", "~"):
            return None
        try:
            return int(s)
        except ValueError:
            pass
        try:
            return float(s)
        except ValueError:
            pass
        if (s.startswith('"') and s.endswith('"')) or (
            s.startswith("'") and s.endswith("'")
        ):
            return s[1:-1]
        return s

    lines = [ln.rstrip() for ln in text.splitlines() if ln.strip() and not ln.lstrip().startswith("#")]
    root: dict = {}
    # Simple two-level support: top-level keys + list-of-mappings under a key.
    i = 0
    while i < len(lines):
        line = lines[i]
        indent = len(line) - len(line.lstrip())
        stripped = line.strip()
        if indent == 0 and ":" in stripped:
            key, _, rest = stripped.partition(":")
            rest = rest.strip()
            if rest == "":
                # Look ahead for a list or nested mapping
                items: list = []
                j = i + 1
                while j < len(lines):
                    nxt = lines[j]
                    nxt_indent = len(nxt) - len(nxt.lstrip())
                    if nxt_indent == 0:
                        break
                    ns = nxt.strip()
                    if ns.startswith("- "):
                        items.append(parse_scalar(ns[2:].strip()))
                    j += 1
                root[key.strip()] = items
                i = j
                continue
            else:
                root[key.strip()] = parse_scalar(rest)
        i += 1
    return root


def _sha8(data: str) -> str:
    return hashlib.sha256(data.encode("utf-8")).hexdigest()[:8]


def _new_proposal_id(agent_name: str) -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{agent_name}-{ts}-{uuid.uuid4().hex[:6]}"


def _load_current_config(writable_keys: list) -> dict:
    """Scan the referenced Python modules and extract current literal values."""
    current: dict = {}
    for entry in writable_keys or []:
        if not isinstance(entry, dict):
            continue
        module = entry.get("module")
        key = entry.get("key")
        default = entry.get("default")
        if not module or not key:
            continue
        path = REPO_ROOT / module
        val: Any = default
        if path.exists():
            try:
                text = path.read_text(encoding="utf-8")
                m = re.search(rf"^\s*{re.escape(key)}\s*=\s*(.+)$", text, re.MULTILINE)
                if m:
                    raw = m.group(1).split("#", 1)[0].strip().rstrip(",")
                    try:
                        val = int(raw)
                    except ValueError:
                        try:
                            val = float(raw)
                        except ValueError:
                            val = raw.strip("'\"")
            except Exception:
                pass
        current[key] = val
    return current


def _stats_snapshot_stub() -> dict:
    """Placeholder stats summary. Real version would read D:/kalshi-data."""
    return {
        "note": "stats snapshot stub — wire real ledger reader in Phase B",
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }


def _extract_proposal_json(raw_text: str) -> Optional[dict]:
    """Pull the <proposal>...</proposal> JSON block out of claude stdout."""
    m = re.search(r"<proposal>(.*?)</proposal>", raw_text, re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(1).strip())
    except json.JSONDecodeError:
        return None


def run_one_agent(
    agent_name: str,
    *,
    mock_codex: bool = False,
    mock_claude: bool = False,
    proposal_root: Optional[Path] = None,
    codex_model: Optional[str] = None,
) -> dict:
    """Execute one strategist agent end-to-end. Returns the proposal dict written."""
    yaml_path = AGENTS_DIR / f"{agent_name}.yaml"
    if not yaml_path.exists():
        raise FileNotFoundError(f"agent YAML not found: {yaml_path}")

    cfg = _load_yaml(yaml_path)
    authority = cfg.get("authority_class", "advisory")
    if authority != "advisory":
        raise ValueError(
            f"agent {agent_name} is authority_class={authority}; orchestrator only runs advisory"
        )

    writable_keys = cfg.get("writable_config_keys") or []
    current_config = _load_current_config(writable_keys)
    stats = _stats_snapshot_stub()

    # Step 1+2+3+4: get a proposal from claude.
    if mock_claude:
        proposal_body = json.loads(json.dumps(MOCK_PROPOSAL_TEMPLATE))  # deep copy
        proposal_body["current_config"] = current_config or proposal_body["current_config"]
        claude_meta = {"runner": "mock"}
    else:
        # Real path — defer to run_claude_headless.run_strategist. We don't
        # exercise this in tests; it requires a real snapshot + claude CLI.
        from scripts.agents.run_claude_headless import run_strategist  # local import

        output_dir = Path(
            os.environ.get("KALSHI_AGENT_WORK_DIR", str(REPO_ROOT / ".agent_work" / agent_name))
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        proposal_body, claude_meta = run_strategist(
            agent_name=agent_name,
            agent_cfg=cfg,
            output_dir=output_dir,
            timeout_s=ORCHESTRATOR_TIMEOUT_S,
        )
        if proposal_body is None:
            raise RuntimeError(f"strategist {agent_name} produced no parseable proposal")

    # Step 5: codex review
    from scripts.agents import codex_review as cr

    verdict = cr.review(
        agent_name=agent_name,
        proposed_config=proposal_body.get("proposed_config", {}),
        current_config=proposal_body.get("current_config", current_config),
        rationale=proposal_body.get("rationale", ""),
        writable_keys=writable_keys,
        stats_snapshot=proposal_body.get("success_criteria_snapshot", stats),
        mock=mock_codex,
        model=codex_model,
    )

    # Step 6: write proposal
    proposal_root = proposal_root or Path(
        os.environ.get("KALSHI_AGENT_PROPOSAL_ROOT", str(DEFAULT_PROPOSAL_ROOT))
    )
    pending_dir = proposal_root / "pending"
    pending_dir.mkdir(parents=True, exist_ok=True)

    pid = _new_proposal_id(agent_name)
    now = datetime.now(timezone.utc).isoformat()

    snapshot_hash = _sha8(json.dumps(stats, sort_keys=True))

    proposal = {
        "id": pid,
        "state": "pending",
        "agent": agent_name,
        "agent_version": cfg.get("version", "0.0.0"),
        "created_at": now,
        "snapshot_hash": snapshot_hash,
        "current_config": proposal_body.get("current_config", current_config),
        "proposed_config": proposal_body.get("proposed_config", {}),
        "rationale": proposal_body.get("rationale", ""),
        "success_criteria_snapshot": proposal_body.get("success_criteria_snapshot", {}),
        "git_patch": proposal_body.get("git_patch", ""),
        "writable_config_keys": writable_keys,
        "codex_review": verdict.to_json(),
        "human_decision": None,
        "applied_at": None,
        "applied_commit": None,
        "claude_meta": claude_meta,
    }

    out_path = pending_dir / f"{pid}.json"
    out_path.write_text(json.dumps(proposal, indent=2), encoding="utf-8")

    # Step 7: telemetry
    _log_usage({
        "timestamp": now,
        "runner": "orchestrator",
        "agent": agent_name,
        "proposal_id": pid,
        "codex_verdict": verdict.verdict,
        "mock_codex": mock_codex,
        "mock_claude": mock_claude,
    })

    print(f"[orchestrator] wrote {out_path}")
    print(f"[orchestrator] codex verdict: {verdict.verdict}")
    return proposal


def _log_usage(event: dict) -> None:
    usage = AGENTS_DIR / ".usage.jsonl"
    usage.parent.mkdir(parents=True, exist_ok=True)
    with usage.open("a", encoding="utf-8") as f:
        f.write(json.dumps(event) + "\n")


def main(argv: Optional[list] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--agent", help="agent name (matches agents/<name>.yaml)")
    ap.add_argument("--all", action="store_true", help="run every advisory agent")
    ap.add_argument("--mock-codex", action="store_true")
    ap.add_argument("--mock-claude", action="store_true")
    ap.add_argument("--model", default=None, help="codex model override")
    args = ap.parse_args(argv)

    if not args.agent and not args.all:
        ap.error("must pass --agent <name> or --all")

    if args.all:
        agents = []
        for p in sorted(AGENTS_DIR.glob("*.yaml")):
            try:
                cfg = _load_yaml(p)
                if cfg.get("authority_class") == "advisory":
                    agents.append(p.stem)
            except Exception:
                continue
    else:
        agents = [args.agent]

    rc = 0
    for name in agents:
        try:
            run_one_agent(
                name,
                mock_codex=args.mock_codex,
                mock_claude=args.mock_claude,
                codex_model=args.model,
            )
        except FileNotFoundError as e:
            print(f"ERROR: {e}", file=sys.stderr)
            rc = 2
        except Exception as e:
            print(f"ERROR: agent {name} failed: {e}", file=sys.stderr)
            rc = 1
    return rc


if __name__ == "__main__":
    sys.exit(main())
