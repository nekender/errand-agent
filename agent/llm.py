"""LLM layer. The orchestrator depends only on the `LLM` protocol, so evals run offline against a
ScriptedLLM (or a ReplayLLM built from recorded real responses) and production uses Claude."""
from __future__ import annotations

import itertools
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Protocol

# USD per 1M tokens: (input, output, cache_read). Used for trace cost estimates.
PRICING = {
    "claude-opus-5-5": (4.00, 20.00, 0.20),
    "claude-sonnet-5-5": (2.00, 10.00, 0.20),
    "claude-haiku-5-5": (0.10, 0.50, 0.01),
    "scripted": (0.0, 0.0, 0.0),
}
_ids = itertools.count(1)


@dataclass
class ToolCall:
    name: str
    args: dict
    id: str = field(default_factory=lambda: f"toolu_{next(_ids):05d}")


@dataclass
class LLMResponse:
    text: str | None
    tool_calls: list[ToolCall]
    assistant_content: list[dict]          # exactly what must be appended to messages (append-only)
    model: str = "scripted"
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    stop_reason: str = "end_turn"
    latency_ms: int = 0

    @property
    def cost_usd(self) -> float:
        pin, pout, pcache = PRICING.get(self.model, PRICING["claude-opus-5-5"])
        uncached = max(self.input_tokens - self.cache_read_tokens, 0)
        return (uncached * pin + self.cache_read_tokens * pcache + self.output_tokens * pout) / 1e6


class LLM(Protocol):
    name: str

    def complete(self, system: str, messages: list[dict], tools: list[dict]) -> LLMResponse: ...


def estimate_tokens(obj: Any) -> int:
    """Cheap, deterministic token estimate (~4 chars/token). Used for the context budget so tests are
    reproducible offline; the real count from the API is recorded alongside it in the trace."""
    s = obj if isinstance(obj, str) else json.dumps(obj, default=str)
    return max(1, len(s) // 4)


def _build_response(text: str | None, calls: list[ToolCall], model: str, **usage) -> LLMResponse:
    content: list[dict] = []
    if text:
        content.append({"type": "text", "text": text})
    for c in calls:
        content.append({"type": "tool_use", "id": c.id, "name": c.name, "input": c.args})
    return LLMResponse(text=text, tool_calls=calls, assistant_content=content, model=model,
                       stop_reason="tool_use" if calls else "end_turn", **usage)


# ----------------------------------------------------------------------------- Claude
class AnthropicLLM:
    def __init__(self, model: str | None = None, effort: str = "medium", max_tokens: int = 4096):
        import anthropic

        self.client = anthropic.Anthropic()
        self.model = model or os.getenv("ERRAND_MODEL", "claude-opus-5-5")
        self.name = self.model
        self.effort = effort
        self.max_tokens = max_tokens

    def complete(self, system: str, messages: list[dict], tools: list[dict]) -> LLMResponse:
        t0 = time.monotonic()
        resp = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
            tools=tools,
            tool_choice={"type": "auto"},
            output_config={"effort": self.effort},
            messages=messages,
        )
        latency = int((time.monotonic() - t0) * 1000)
        if resp.stop_reason == "refusal":
            raise RuntimeError("model refused the request")
        text_parts, calls = [], []
        for block in resp.content:
            if block.type == "text":
                text_parts.append(block.text)
            elif block.type == "tool_use":
                calls.append(ToolCall(name=block.name, args=dict(block.input), id=block.id))
        u = resp.usage
        return LLMResponse(
            text="\n".join(text_parts).strip() or None,
            tool_calls=calls,
            assistant_content=[b.model_dump(exclude_none=True) for b in resp.content],
            model=self.model,
            input_tokens=u.input_tokens + (u.cache_read_input_tokens or 0) + (u.cache_creation_input_tokens or 0),
            output_tokens=u.output_tokens,
            cache_read_tokens=u.cache_read_input_tokens or 0,
            stop_reason=resp.stop_reason,
            latency_ms=latency,
        )


# ----------------------------------------------------------------------------- offline
Step = dict | Callable[[str, list[dict], list[dict]], dict]


def call(name: str, **args) -> dict:
    """Script step: the model calls one tool."""
    return {"tool_calls": [{"name": name, "args": args}]}


def say(text: str) -> dict:
    """Script step: the model replies with text and ends its turn."""
    return {"text": text}


class ScriptExhausted(Exception):
    pass


class ScriptedLLM:
    """Deterministic stand-in for the model. Each step is a dict ({"text"} / {"tool_calls"}) or a
    callable(system, messages, tools) -> dict, so a script can react to tool results."""

    name = "scripted"

    def __init__(self, steps: list[Step], default: Step | None = None):
        self.steps = list(steps)
        self.default = default
        self.requests: list[dict] = []   # every request we received, for assertions

    def complete(self, system: str, messages: list[dict], tools: list[dict]) -> LLMResponse:
        self.requests.append({"system": system, "messages": json.loads(json.dumps(messages, default=str)),
                              "tools": [t["name"] for t in tools]})
        if self.steps:
            step = self.steps.pop(0)
        elif self.default is not None:
            step = self.default
        else:
            raise ScriptExhausted("scripted model has no more steps")
        if callable(step):
            step = step(system, messages, tools)
        calls = [ToolCall(name=c["name"], args=c.get("args", {})) for c in step.get("tool_calls", [])]
        return _build_response(step.get("text"), calls, "scripted",
                               input_tokens=estimate_tokens(system) + estimate_tokens(messages),
                               output_tokens=estimate_tokens(step))


class RecordingLLM:
    """Wraps a real LLM and saves normalized responses so they can be replayed offline in evals."""

    def __init__(self, inner: LLM, path: str | Path):
        self.inner, self.path, self.name = inner, Path(path), inner.name
        self.recorded: list[dict] = []

    def complete(self, system: str, messages: list[dict], tools: list[dict]) -> LLMResponse:
        r = self.inner.complete(system, messages, tools)
        self.recorded.append({"text": r.text, "tool_calls": [{"name": c.name, "args": c.args} for c in r.tool_calls],
                              "usage": {"input_tokens": r.input_tokens, "output_tokens": r.output_tokens}})
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.recorded, indent=2, ensure_ascii=False))
        return r


class ReplayLLM(ScriptedLLM):
    """Replays a RecordingLLM fixture."""

    def __init__(self, path: str | Path):
        super().__init__([{k: v for k, v in s.items() if k != "usage"} for s in json.loads(Path(path).read_text())])
