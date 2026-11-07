"""Hard execution budgets. Every limit breach raises BudgetExceeded, which the orchestrator turns
into a clear user-facing outcome."""
from __future__ import annotations

import time
from dataclasses import dataclass, field


class BudgetExceeded(Exception):
    def __init__(self, which: str, limit: float):
        super().__init__(f"{which} limit reached ({limit})")
        self.which = which
        self.limit = limit


@dataclass
class Limits:
    max_model_calls: int = 8        # per user turn
    max_tool_calls: int = 16        # per user turn
    wall_clock_s: float = 90.0      # per user turn
    max_tool_retries: int = 3       # per tool invocation (transient errors)
    max_consecutive_tool_errors: int = 4
    retry_backoff_s: float = 0.0    # tests use 0; CLI may raise it


@dataclass
class Budget:
    limits: Limits
    started: float = field(default_factory=time.monotonic)
    model_calls: int = 0
    tool_calls: int = 0
    consecutive_errors: int = 0

    def check_time(self) -> None:
        if time.monotonic() - self.started > self.limits.wall_clock_s:
            raise BudgetExceeded("wall_clock_s", self.limits.wall_clock_s)

    def model_call(self) -> None:
        self.check_time()
        if self.model_calls >= self.limits.max_model_calls:
            raise BudgetExceeded("max_model_calls", self.limits.max_model_calls)
        self.model_calls += 1

    def tool_call(self) -> None:
        self.check_time()
        if self.tool_calls >= self.limits.max_tool_calls:
            raise BudgetExceeded("max_tool_calls", self.limits.max_tool_calls)
        self.tool_calls += 1

    def tool_error(self) -> None:
        self.consecutive_errors += 1
        if self.consecutive_errors >= self.limits.max_consecutive_tool_errors:
            raise BudgetExceeded("max_consecutive_tool_errors", self.limits.max_consecutive_tool_errors)

    def tool_ok(self) -> None:
        self.consecutive_errors = 0
