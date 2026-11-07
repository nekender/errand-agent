"""JSONL tracing: one file per run (session). Every event carries run/turn ids and a timestamp."""
from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any

# Phone numbers always carry a +country prefix in our data; requiring it keeps ISO dates/times intact.
PHONE_RE = re.compile(r"\+\d{1,3}[\- ]?\d{5}[\- ]?\d{3,5}\b")
MAX_STR = 400


def sanitize(obj: Any) -> Any:
    """Redact phone numbers and truncate long strings so traces are safe to share."""
    if isinstance(obj, str):
        s = PHONE_RE.sub("[redacted-phone]", obj)
        return s if len(s) <= MAX_STR else s[:MAX_STR] + f"…(+{len(s) - MAX_STR} chars)"
    if isinstance(obj, dict):
        return {k: ("[redacted]" if k == "phone" else sanitize(v)) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [sanitize(v) for v in obj]
    return obj


class Tracer:
    def __init__(self, run_id: str, out_dir: str | Path | None = "traces"):
        self.run_id = run_id
        self.events: list[dict] = []
        self.turn = 0
        self.path = None
        if out_dir is not None:
            Path(out_dir).mkdir(parents=True, exist_ok=True)
            self.path = Path(out_dir) / f"{run_id}.jsonl"

    def emit(self, kind: str, **data: Any) -> dict:
        ev = {"ts": round(time.time(), 3), "run_id": self.run_id, "turn": self.turn, "kind": kind,
              **sanitize(data)}
        self.events.append(ev)
        if self.path:
            with self.path.open("a") as f:
                f.write(json.dumps(ev, default=str) + "\n")
        return ev

    def of_kind(self, kind: str) -> list[dict]:
        return [e for e in self.events if e["kind"] == kind]

    def totals(self) -> dict:
        calls = self.of_kind("model_call")
        return {
            "model_calls": len(calls),
            "tool_calls": len(self.of_kind("tool_call")),
            "input_tokens": sum(c.get("input_tokens", 0) for c in calls),
            "output_tokens": sum(c.get("output_tokens", 0) for c in calls),
            "cost_usd": round(sum(c.get("cost_usd", 0.0) for c in calls), 6),
        }
