"""Codex reviewer for agent config proposals.

Uses `codex exec` (not CCB MCP). Zero API key — codex CLI uses its own OAuth.
Returns structured {verdict, reasoning, suggested_revision} via JSON schema.
"""
from __future__ import annotations

import json
import subprocess
import tempfile
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


@dataclass
class CodexVerdict:
    verdict: str                      # "approve" | "reject" | "revise" | "review_failed"
    reasoning: str
    suggested_revision: Optional[dict] = None
    reviewed_at: str = ""
    model: str = ""
    exit_code: int = 0
    elapsed_s: float = 0.0

    def to_json(self) -> dict:
        return asdict(self)


# JSON schema codex must produce. MUST have additionalProperties: false.
VERDICT_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["verdict", "reasoning"],
    "properties": {
        "verdict": {"type": "string", "enum": ["approve", "reject", "revise"]},
        "reasoning": {"type": "string"},
        "suggested_revision": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "proposed_config": {"type": "object", "additionalProperties": True},
                "note": {"type": "string"},
            },
        },
    },
}


def _build_review_prompt(
    *,
    agent_name: str,
    proposed_config: dict,
    current_config: dict,
    rationale: str,
    writable_keys: list,
    stats_snapshot: dict,
) -> str:
    """Compose the review prompt from proposal context."""
    return f"""You are reviewing a config tweak proposal from an automated trading strategy agent for the kalshi-trading-bot.

AGENT: {agent_name}

WRITABLE KEYS (keys this agent is authorized to propose changes to):
{json.dumps(writable_keys, indent=2)}

CURRENT LIVE CONFIG:
{json.dumps(current_config, indent=2)}

PROPOSED CONFIG:
{json.dumps(proposed_config, indent=2)}

AGENT'S RATIONALE:
{rationale}

RECENT PERFORMANCE STATS (rolling 7-day window from ledger/journal):
{json.dumps(stats_snapshot, indent=2)}

YOUR TASK: decide if this proposal should be approved, rejected, or revised. Respond ONLY with JSON matching the provided schema. Criteria:
- Approve if the proposal is within the writable keys, the rationale is sound, the stats support the change, and the risk is bounded.
- Reject if the proposal touches keys it shouldn't, the rationale is wrong, the stats contradict it, or the risk is unbounded.
- Revise if the direction is right but a specific parameter should be different — include suggested_revision.proposed_config with your counter-proposal.

Be terse. Reasoning should be <=200 words. Do not include any text outside the JSON.
"""


def review(
    *,
    agent_name: str,
    proposed_config: dict,
    current_config: dict,
    rationale: str,
    writable_keys: list,
    stats_snapshot: dict,
    timeout_s: int = 600,
    mock: bool = False,
    model: Optional[str] = None,
) -> CodexVerdict:
    """Run codex exec to review a proposal. Returns a CodexVerdict."""
    if mock:
        return CodexVerdict(
            verdict="approve",
            reasoning="[MOCK MODE] auto-approved for testing",
            reviewed_at=datetime.now(timezone.utc).isoformat(),
            model="mock",
        )

    prompt = _build_review_prompt(
        agent_name=agent_name,
        proposed_config=proposed_config,
        current_config=current_config,
        rationale=rationale,
        writable_keys=writable_keys,
        stats_snapshot=stats_snapshot,
    )

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        schema_path = tmp_path / "schema.json"
        schema_path.write_text(json.dumps(VERDICT_SCHEMA))
        output_path = tmp_path / "verdict.json"

        cmd = [
            "codex", "exec",
            "--json",
            "--skip-git-repo-check",
            "--sandbox", "read-only",
            "--output-schema", str(schema_path),
            "--output-last-message", str(output_path),
        ]
        if model:
            cmd += ["-m", model]
        cmd.append(prompt)

        started = datetime.now(timezone.utc)
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s)
        except subprocess.TimeoutExpired:
            return CodexVerdict(
                verdict="review_failed",
                reasoning=f"codex exec timed out after {timeout_s}s",
                reviewed_at=started.isoformat(),
                exit_code=124,
                elapsed_s=float(timeout_s),
            )
        except FileNotFoundError:
            return CodexVerdict(
                verdict="review_failed",
                reasoning="codex CLI not found on PATH",
                reviewed_at=started.isoformat(),
                exit_code=127,
                elapsed_s=0.0,
            )

        elapsed = (datetime.now(timezone.utc) - started).total_seconds()

        if proc.returncode != 0 or not output_path.exists():
            return CodexVerdict(
                verdict="review_failed",
                reasoning=f"codex exec exit {proc.returncode}: {(proc.stderr or '')[-400:]}",
                reviewed_at=started.isoformat(),
                exit_code=proc.returncode,
                elapsed_s=elapsed,
            )

        try:
            raw = output_path.read_text().strip()
            if not raw:
                return CodexVerdict(
                    verdict="review_failed",
                    reasoning="codex produced empty output",
                    reviewed_at=started.isoformat(),
                    elapsed_s=elapsed,
                )
            parsed = json.loads(raw)
        except json.JSONDecodeError as e:
            return CodexVerdict(
                verdict="review_failed",
                reasoning=f"codex output not JSON: {e}",
                reviewed_at=started.isoformat(),
                elapsed_s=elapsed,
            )

        return CodexVerdict(
            verdict=parsed.get("verdict", "review_failed"),
            reasoning=parsed.get("reasoning", ""),
            suggested_revision=parsed.get("suggested_revision"),
            reviewed_at=started.isoformat(),
            model=model or "codex-default",
            elapsed_s=elapsed,
        )
