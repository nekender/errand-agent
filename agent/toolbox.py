"""Single choke point for every contract-tool call: tracing, latency, budget, bounded retries."""
from __future__ import annotations

import time

from agent.limits import Budget
from mocks.tools import ToolError, ToolTimeout


class Toolbox:
    def __init__(self, world, tracer, limits):
        self.world, self.tracer, self.limits = world, tracer, limits
        self.budget: Budget | None = None   # set per turn by the orchestrator

    def call(self, name: str, *, retry_transient: bool = True, **args):
        """Call a contract tool. Transient errors (503) are retried with the *same* arguments (including
        idempotency keys) up to max_tool_retries. Timeouts are NOT retried here when retry_transient=False:
        for side effects the caller must consult the source of truth first."""
        attempt = 0
        while True:
            attempt += 1
            if self.budget:
                self.budget.tool_call()
            t0 = time.monotonic()
            try:
                result = getattr(self.world, name)(**args)
            except ToolError as e:
                ms = int((time.monotonic() - t0) * 1000)
                self.tracer.emit("tool_call", name=name, args=args, error=e.code, message=str(e),
                                 attempt=attempt, latency_ms=ms)
                if self.budget:
                    self.budget.tool_error()
                if isinstance(e, ToolTimeout) and not retry_transient:
                    raise
                if e.retryable and retry_transient and attempt <= self.limits.max_tool_retries:
                    time.sleep(self.limits.retry_backoff_s * attempt)
                    continue
                raise
            ms = int((time.monotonic() - t0) * 1000)
            self.tracer.emit("tool_call", name=name, args=args, result=result, attempt=attempt, latency_ms=ms)
            if self.budget:
                self.budget.tool_ok()
            return result
