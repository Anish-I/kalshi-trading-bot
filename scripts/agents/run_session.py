"""Run a Managed Agents session end-to-end.

- Uploads (or inlines) a snapshot tarball
- Streams SSE events
- Extracts <tarball>...</tarball> from the final assistant message
- Untars into output_dir
- Appends token/cost telemetry to agents/.usage.jsonl

ANTHROPIC_API_KEY must be set.
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import os
import re
import sys
import tarfile
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
AGENTS_DIR = REPO_ROOT / "agents"
DEPLOYED_PATH = AGENTS_DIR / ".deployed.json"
USAGE_PATH = AGENTS_DIR / ".usage.jsonl"

TARBALL_RE = re.compile(r"<tarball>\s*(.*?)\s*</tarball>", re.DOTALL)


def _load_deployed() -> dict:
    if DEPLOYED_PATH.exists():
        try:
            return json.loads(DEPLOYED_PATH.read_text())
        except Exception:
            return {}
    return {}


def _append_usage(row: dict) -> None:
    USAGE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with USAGE_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")


def _extract_text_from_event(event) -> str:
    """Best-effort extraction of assistant text from an SSE event object."""
    parts: list[str] = []
    data = getattr(event, "data", None) or getattr(event, "message", None) or event
    try:
        content = getattr(data, "content", None)
        if content is None and isinstance(data, dict):
            content = data.get("content")
        if isinstance(content, list):
            for block in content:
                t = getattr(block, "text", None)
                if t is None and isinstance(block, dict):
                    t = block.get("text")
                if t:
                    parts.append(t)
        elif isinstance(content, str):
            parts.append(content)
    except Exception:
        pass
    return "".join(parts)


def _extract_tarball(final_text: str, output_dir: Path) -> bool:
    m = TARBALL_RE.search(final_text or "")
    if not m:
        return False
    b64 = re.sub(r"\s+", "", m.group(1))
    try:
        raw = base64.b64decode(b64)
    except Exception as e:
        print(f"base64 decode failed: {e}", file=sys.stderr)
        return False
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        with tarfile.open(fileobj=io.BytesIO(raw), mode="r:gz") as tar:
            tar.extractall(output_dir)
    except Exception as e:
        print(f"tar extract failed: {e}", file=sys.stderr)
        return False
    return True


def run(
    agent_name: str,
    tarball_path: Path,
    output_dir: Path,
    timeout_s: int = 600,
) -> int:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print(
            "ANTHROPIC_API_KEY not set; cannot run a live session",
            file=sys.stderr,
        )
        return 2

    try:
        import anthropic  # type: ignore
    except ImportError:
        print("anthropic SDK not installed", file=sys.stderr)
        return 2

    deployed = _load_deployed()
    entry = deployed.get(agent_name)
    if not entry or not entry.get("id"):
        print(
            f"agent {agent_name} not in {DEPLOYED_PATH}; run deploy_agents.py first",
            file=sys.stderr,
        )
        return 3

    client = anthropic.Anthropic()
    tarball_path = Path(tarball_path)
    if not tarball_path.exists():
        print(f"tarball not found: {tarball_path}", file=sys.stderr)
        return 3

    # ---- file attachment strategy ----
    # Strategy 1: files.upload -> reference file id in the user turn
    # Strategy 2: inline base64 in the user message
    file_id = None
    attachment_strategy = "inline-base64"
    try:
        with tarball_path.open("rb") as fh:
            uploaded = client.beta.files.upload(file=fh)  # type: ignore[attr-defined]
        file_id = getattr(uploaded, "id", None)
        if file_id:
            attachment_strategy = "files.upload"
    except Exception as e:
        print(
            f"files.upload failed ({type(e).__name__}: {e}); falling back to inline base64",
            file=sys.stderr,
        )

    if attachment_strategy == "inline-base64":
        b64 = base64.b64encode(tarball_path.read_bytes()).decode()
        user_text = (
            "Generate the daily report per your system prompt. "
            "The inputs tarball is inline below. As your first bash call, run:\n"
            "  mkdir -p /work/inputs && "
            'printf "%s" "$INPUT_TARBALL" | base64 -d | tar xzC /work/inputs/\n'
            "where $INPUT_TARBALL is the content between the <input_tarball> tags.\n"
            f"<input_tarball>{b64}</input_tarball>"
        )
    else:
        user_text = (
            "Generate the daily report per your system prompt. "
            f"Inputs tarball attached as file id {file_id}. "
            "Unpack it into /work/inputs/ as your first bash call."
        )

    # ---- environment ----
    try:
        env = client.beta.environments.create(
            config={"type": "cloud", "networking": {"type": "unrestricted"}}
        )
        env_id = getattr(env, "id", None)
    except Exception as e:
        print(f"environment create failed: {type(e).__name__}: {e}", file=sys.stderr)
        return 3

    # ---- session ----
    session_kwargs = {
        "agent_id": entry["id"],
        "environment_id": env_id,
    }
    try:
        session = client.beta.sessions.create(**session_kwargs)
        session_id = getattr(session, "id", None)
    except Exception as e:
        print(f"sessions.create failed: {type(e).__name__}: {e}", file=sys.stderr)
        return 3

    # ---- send initial user message ----
    try:
        client.beta.sessions.events.send(
            session_id,
            role="user",
            content=[{"type": "text", "text": user_text}],
        )
    except Exception as e:
        print(
            f"sessions.events.send failed: {type(e).__name__}: {e}", file=sys.stderr
        )
        return 3

    # ---- stream events ----
    started = time.time()
    final_text = ""
    input_tokens = 0
    output_tokens = 0
    try:
        stream = client.beta.sessions.events.stream(session_id)
        for event in stream:
            now = time.time()
            if now - started > timeout_s:
                print(f"timeout {timeout_s}s exceeded; cancelling", file=sys.stderr)
                try:
                    client.beta.sessions.archive(session_id)
                except Exception:
                    pass
                return 4
            ev_type = getattr(event, "type", type(event).__name__)
            ev_name = getattr(event, "name", "")
            print(f"[{now - started:7.1f}s] {ev_type} {ev_name}".rstrip())
            # Try to accumulate usage and final text
            usage = getattr(event, "usage", None)
            if usage:
                input_tokens = int(getattr(usage, "input_tokens", input_tokens) or 0)
                output_tokens = int(getattr(usage, "output_tokens", output_tokens) or 0)
            text = _extract_text_from_event(event)
            if text:
                final_text = text  # keep the latest assistant text block
    except Exception as e:
        print(f"stream error: {type(e).__name__}: {e}", file=sys.stderr)
        elapsed = time.time() - started
        _append_usage(
            {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "agent": agent_name,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "elapsed_s": round(elapsed, 2),
                "exit_code": 5,
            }
        )
        return 5

    elapsed = time.time() - started
    ok = _extract_tarball(final_text, Path(output_dir))
    exit_code = 0 if ok else 5
    if not ok:
        print("no tarball in final message", file=sys.stderr)

    _append_usage(
        {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "agent": agent_name,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "elapsed_s": round(elapsed, 2),
            "exit_code": exit_code,
        }
    )
    return exit_code


def _main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--agent", required=True)
    ap.add_argument("--tarball", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--timeout", type=int, default=600)
    args = ap.parse_args()
    return run(args.agent, Path(args.tarball), Path(args.out), timeout_s=args.timeout)


if __name__ == "__main__":
    sys.exit(_main())
