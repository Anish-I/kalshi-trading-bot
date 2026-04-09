"""Deploy Claude Managed Agents from agents/*.yaml configs.

Usage:
    python scripts/agents/deploy_agents.py             # create/update
    python scripts/agents/deploy_agents.py --dry-run   # print payloads, no API calls

Requires: pip install pyyaml anthropic
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

try:
    import yaml  # type: ignore
    HAVE_YAML = True
except ImportError:
    HAVE_YAML = False

REPO_ROOT = Path(__file__).resolve().parents[2]
AGENTS_DIR = REPO_ROOT / "agents"
DEPLOYED_PATH = AGENTS_DIR / ".deployed.json"


def _tiny_yaml_load(text: str) -> dict:
    """Minimal fallback for flat top-level keys + `tools:` list of mappings."""
    out: dict = {}
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i].rstrip()
        if not line or line.lstrip().startswith("#"):
            i += 1
            continue
        if line.startswith("tools:"):
            tools = []
            i += 1
            current: dict = {}
            while i < len(lines) and (lines[i].startswith(" ") or lines[i].startswith("\t") or lines[i].startswith("-")):
                raw = lines[i]
                stripped = raw.strip()
                if stripped.startswith("- "):
                    if current:
                        tools.append(current)
                    current = {}
                    kv = stripped[2:]
                    if ":" in kv:
                        k, v = kv.split(":", 1)
                        current[k.strip()] = v.strip()
                elif ":" in stripped:
                    k, v = stripped.split(":", 1)
                    current[k.strip()] = v.strip()
                i += 1
            if current:
                tools.append(current)
            out["tools"] = tools
            continue
        if ":" in line:
            k, v = line.split(":", 1)
            out[k.strip()] = v.strip()
        i += 1
    return out


def load_yaml(path: Path) -> dict:
    text = path.read_text()
    if HAVE_YAML:
        return yaml.safe_load(text)
    return _tiny_yaml_load(text)


def build_system_prompt(cfg: dict) -> str:
    header = (AGENTS_DIR / cfg["common_header_path"]).read_text()
    body = (AGENTS_DIR / cfg["system_prompt_path"]).read_text()
    return header.strip() + "\n\n" + body.strip() + "\n"


def build_payload(cfg: dict) -> dict:
    return {
        "name": cfg["name"],
        "model": cfg["model"],
        "system": build_system_prompt(cfg),
        "tools": cfg.get("tools", []),
        "version": cfg.get("version", "0.0.0"),
    }


def _load_deployed() -> dict:
    if DEPLOYED_PATH.exists():
        try:
            return json.loads(DEPLOYED_PATH.read_text())
        except Exception:
            return {}
    return {}


def _save_deployed(data: dict) -> None:
    DEPLOYED_PATH.write_text(json.dumps(data, indent=2))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    yamls = sorted(AGENTS_DIR.glob("*.yaml"))
    if not yamls:
        print("no agent YAML files found under agents/", file=sys.stderr)
        return 1

    configs = [load_yaml(p) for p in yamls]
    payloads = [build_payload(c) for c in configs]

    if args.dry_run:
        print(json.dumps(payloads, indent=2))
        return 0

    if not os.environ.get("ANTHROPIC_API_KEY"):
        print(
            "ANTHROPIC_API_KEY not set; use --dry-run or export the key",
            file=sys.stderr,
        )
        return 2

    try:
        import anthropic  # type: ignore
    except ImportError:
        print("anthropic SDK not installed (pip install anthropic)", file=sys.stderr)
        return 2

    client = anthropic.Anthropic()
    deployed = _load_deployed()

    for p in payloads:
        name = p["name"]
        entry = deployed.get(name)
        kwargs = {
            "name": p["name"],
            "model": p["model"],
            "system": p["system"],
            "tools": p["tools"],
        }
        try:
            if entry and entry.get("id"):
                result = client.beta.agents.update(entry["id"], **kwargs)
                action = "updated"
            else:
                result = client.beta.agents.create(**kwargs)
                action = "created"
            agent_id = getattr(result, "id", None) or (
                result.get("id") if isinstance(result, dict) else None
            )
            deployed[name] = {
                "id": agent_id,
                "version": p["version"],
                "deployed_at": datetime.now(timezone.utc).isoformat(),
            }
            print(f"{action} agent {name} -> {agent_id}")
        except Exception as e:
            print(f"failed to deploy {name}: {type(e).__name__}: {e}", file=sys.stderr)
            _save_deployed(deployed)
            return 1

    _save_deployed(deployed)
    return 0


if __name__ == "__main__":
    sys.exit(main())
