"""Rule-based simulated user. Reacts to what the agent actually says (e.g. approves only when shown a
confirmation that matches its expectations), so scenarios exercise the real control flow."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable


@dataclass
class SimUser:
    opening: str
    # (pattern on agent reply, response or callable(reply) -> response). First match wins.
    rules: list[tuple[str, str | Callable[[str], str]]] = field(default_factory=list)
    max_turns: int = 8
    transcript: list[tuple[str, str]] = field(default_factory=list)

    def respond(self, reply: str) -> str | None:
        for pat, resp in self.rules:
            if re.search(pat, reply, re.I):
                return resp(reply) if callable(resp) else resp
        return None

    def run(self, agent) -> list[tuple[str, str]]:
        msg: str | None = self.opening
        for _ in range(self.max_turns):
            if msg is None:
                break
            reply = agent.handle(msg)
            self.transcript.append((msg, reply))
            msg = self.respond(reply)
        return self.transcript


def approve_if(max_price: int | None = None, min_hour: int | None = None):
    """A cautious user: says yes only if the confirmation meets their constraints, otherwise no."""
    def _decide(reply: str) -> str:
        price = re.search(r"₹([\d,]+)\.? The slot|for ₹([\d,]+)", reply)
        hour = re.search(r", (\d{1,2}):\d\d (AM|PM)", reply)
        if max_price and price:
            p = int((price.group(1) or price.group(2)).replace(",", ""))
            if p > max_price:
                return "no"
        if min_hour is not None and hour:
            h = int(hour.group(1)) % 12 + (12 if hour.group(2) == "PM" else 0)
            if h < min_hour:
                return "no"
        return "yes"
    return _decide
