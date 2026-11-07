"""Mock implementations of the 8 fixed tool contracts.

The MockWorld plays the role of the external provider platform. It is the *source of truth*
for holds and bookings; the agent only sees it through the 8 contract methods. Failure
behaviour is driven by a seeded RNG, and every outcome can be forced via `Scenario` so
tests are reproducible.
"""
from __future__ import annotations

import copy
import hashlib
import json
import random
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

from mocks.world_seed import DEFAULT_NOW, build_catalog, seed_bookings

HOLD_TTL = timedelta(minutes=10)


class ToolError(Exception):
    """Transport-level failure (not a business error). Business errors are returned as {"error": ...}."""

    code = "tool_error"
    retryable = False


class ToolTimeout(ToolError):
    code = "timeout"
    retryable = True


class ToolUnavailable(ToolError):
    code = "503_service_unavailable"
    retryable = True


class Clock:
    """Controllable clock so hold expiry is testable."""

    def __init__(self, now: datetime = DEFAULT_NOW):
        self._now = now

    def now(self) -> datetime:
        return self._now

    def advance(self, **kw) -> None:
        self._now += timedelta(**kw)


@dataclass
class Scenario:
    seed: int = 42
    # Forced outcomes for successive book() calls: "success" | "slot_taken" | "timeout_booked" | "timeout_not_booked"
    book_outcomes: list[str] = field(default_factory=list)
    # Added to the quoted price at hold time, keyed by provider_id ("*" = all providers).
    hold_price_delta: dict[str, int] = field(default_factory=dict)
    # Slots (ISO, minute precision) that reschedule() treats as conflicting.
    conflict_slots: list[str] = field(default_factory=list)
    # Tool name -> number of calls that should raise ToolError("unexpected") before working.
    tool_crashes: dict[str, int] = field(default_factory=dict)
    # Hook: called after book() records a booking but before it returns (used to simulate a crash).
    after_book_commit: Any = None


def _key(*parts: Any) -> str:
    return hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()[:16]


class MockWorld:
    def __init__(self, scenario: Scenario | None = None, clock: Clock | None = None,
                 persist_path: str | Path | None = None):
        self.scenario = scenario or Scenario()
        self.rng = random.Random(self.scenario.seed)
        self.clock = clock or Clock()
        self.catalog = build_catalog()
        self.bookings: dict[str, dict] = {b["booking_id"]: b for b in seed_bookings()}
        self.holds: dict[str, dict] = {}
        self.idem: dict[str, dict] = {}            # idempotency_key -> stored result (server-side dedupe)
        self.cancel_attempts: dict[str, int] = {}  # booking_id -> attempts (first one 503s)
        self.calls: list[dict] = []                # audit log of every contract call
        self._crash_budget = dict(self.scenario.tool_crashes)
        self._counter = 0
        self.persist_path = Path(persist_path) if persist_path else None
        if self.persist_path and self.persist_path.exists():
            self._load()

    # ------------------------------------------------------------------ helpers
    def _next_id(self, prefix: str) -> str:
        self._counter += 1
        return f"{prefix}_{self._counter:04d}"

    def _log(self, name: str, args: dict) -> None:
        self.calls.append({"tool": name, "args": args, "at": self.clock.now().isoformat()})
        if self._crash_budget.get(name, 0) > 0:
            self._crash_budget[name] -= 1
            raise ToolError(f"unexpected internal error in {name}")

    def _save(self) -> None:
        if not self.persist_path:
            return
        state = {k: getattr(self, k) for k in ("bookings", "holds", "idem", "cancel_attempts", "_counter")}
        self.persist_path.write_text(json.dumps(state, default=str))

    def _load(self) -> None:
        state = json.loads(self.persist_path.read_text())
        for k, v in state.items():
            setattr(self, k, v)

    def _slots_for(self, p: dict, start: date, end: date) -> list[str]:
        out, d = [], start
        while d <= end:
            # Provider-specific deterministic availability: skip some days.
            if int(_key(p["provider_id"], d.isoformat()), 16) % 5 != 0:
                for h in p["hours"]:
                    slot = datetime(d.year, d.month, d.day, h)
                    if slot > self.clock.now() and not self._slot_taken(p["provider_id"], slot.isoformat(timespec="minutes")):
                        out.append(slot.isoformat(timespec="minutes"))
            d += timedelta(days=1)
        return out

    def _slot_taken(self, provider_id: str, slot: str) -> bool:
        return any(b["provider_id"] == provider_id and b["slot"] == slot and b["status"] == "confirmed"
                   for b in self.bookings.values())

    # ------------------------------------------------------------- contracts
    def search_providers(self, service: str, area: str, date_range: dict) -> dict:
        self._log("search_providers", {"service": service, "area": area, "date_range": date_range})
        start = date.fromisoformat(date_range["start"])
        end = date.fromisoformat(date_range["end"])
        pool = [p for p in self.catalog.values() if p["service"] == service]
        if not pool:
            return {"providers": [], "error": "unknown_service"}
        same_area = [p for p in pool if p["area"].lower() == (area or "").lower()]
        others = [p for p in pool if p not in same_area]
        self.rng.shuffle(others)
        n = self.rng.randint(3, 8)
        chosen = (same_area + others)[:n]
        results = []
        for p in chosen:
            results.append({
                "provider_id": p["provider_id"], "name": p["name"], "area": p["area"],
                "price": p["price"], "rating": p["rating"], "attributes": p["attributes"],
                "available_slots": self._slots_for(p, start, end)[:12],
            })
        return {"providers": results}

    def get_provider(self, provider_id: str) -> dict:
        self._log("get_provider", {"provider_id": provider_id})
        p = self.catalog.get(provider_id)
        if not p:
            return {"error": "not_found"}
        return {k: copy.deepcopy(v) for k, v in p.items() if k != "hours"}

    def hold_slot(self, provider_id: str, slot: str) -> dict:
        self._log("hold_slot", {"provider_id": provider_id, "slot": slot})
        p = self.catalog.get(provider_id)
        if not p:
            return {"error": "not_found"}
        slot_dt = datetime.fromisoformat(slot)
        if slot_dt.hour not in p["hours"] or slot_dt <= self.clock.now() or self._slot_taken(provider_id, slot):
            return {"error": "slot_unavailable"}
        delta = self.scenario.hold_price_delta.get(provider_id, self.scenario.hold_price_delta.get("*", 0))
        hold_id = self._next_id("hold")
        hold = {"hold_id": hold_id, "provider_id": provider_id, "slot": slot,
                "quoted_price": p["price"] + delta, "expires_at": (self.clock.now() + HOLD_TTL).isoformat()}
        self.holds[hold_id] = hold
        self._save()
        return dict(hold)

    def book(self, hold_id: str, idempotency_key: str) -> dict:
        self._log("book", {"hold_id": hold_id, "idempotency_key": idempotency_key})
        if idempotency_key in self.idem:                       # server-side idempotency
            return dict(self.idem[idempotency_key])
        hold = self.holds.get(hold_id)
        if not hold:
            return {"error": "hold_not_found"}
        if datetime.fromisoformat(hold["expires_at"]) <= self.clock.now():
            return {"error": "hold_expired"}

        if self.scenario.book_outcomes:
            outcome = self.scenario.book_outcomes.pop(0)
        else:
            r = self.rng.random()
            outcome = ("success" if r < 0.70 else "slot_taken" if r < 0.85
                       else "timeout_booked" if r < 0.925 else "timeout_not_booked")

        if outcome == "slot_taken":
            return {"error": "slot_taken"}
        if outcome == "timeout_not_booked":
            raise ToolTimeout("book timed out")

        booking_id = self._next_id("bk")
        booking = {"booking_id": booking_id, "user_id": "u1", "provider_id": hold["provider_id"],
                   "service": self.catalog[hold["provider_id"]]["service"], "slot": hold["slot"],
                   "price": hold["quoted_price"], "status": "confirmed",
                   "hold_id": hold_id, "idempotency_key": idempotency_key}
        self.bookings[booking_id] = booking
        result = {"booking_id": booking_id, "status": "confirmed", "slot": hold["slot"],
                  "price": hold["quoted_price"], "provider_id": hold["provider_id"]}
        self.idem[idempotency_key] = result
        self._save()
        if self.scenario.after_book_commit:
            self.scenario.after_book_commit(result)
        if outcome == "timeout_booked":
            raise ToolTimeout("book timed out (but booking was committed)")
        return dict(result)

    def get_booking_status(self, hold_id: str | None = None, idempotency_key: str | None = None) -> dict:
        self._log("get_booking_status", {"hold_id": hold_id, "idempotency_key": idempotency_key})
        for b in self.bookings.values():
            if (idempotency_key and b.get("idempotency_key") == idempotency_key) or \
               (hold_id and b.get("hold_id") == hold_id):
                return {"status": b["status"], "booking_id": b["booking_id"], "slot": b["slot"],
                        "price": b["price"], "provider_id": b["provider_id"]}
        if hold_id and hold_id in self.holds:
            h = self.holds[hold_id]
            expired = datetime.fromisoformat(h["expires_at"]) <= self.clock.now()
            return {"status": "hold_expired" if expired else "held", "booking_id": None}
        return {"status": "not_found", "booking_id": None}

    def cancel(self, booking_id: str, idempotency_key: str) -> dict:
        self._log("cancel", {"booking_id": booking_id, "idempotency_key": idempotency_key})
        if idempotency_key in self.idem:
            return dict(self.idem[idempotency_key])
        b = self.bookings.get(booking_id)
        if not b:
            return {"error": "not_found"}
        self.cancel_attempts[booking_id] = self.cancel_attempts.get(booking_id, 0) + 1
        if self.cancel_attempts[booking_id] == 1:
            self._save()
            raise ToolUnavailable("503 Service Unavailable")
        b["status"] = "cancelled"
        result = {"booking_id": booking_id, "status": "cancelled"}
        self.idem[idempotency_key] = result
        self._save()
        return dict(result)

    def reschedule(self, booking_id: str, slot: str, idempotency_key: str) -> dict:
        self._log("reschedule", {"booking_id": booking_id, "slot": slot, "idempotency_key": idempotency_key})
        if idempotency_key in self.idem:
            return dict(self.idem[idempotency_key])
        b = self.bookings.get(booking_id)
        if not b or b["status"] != "confirmed":
            return {"error": "not_found"}
        p = self.catalog[b["provider_id"]]
        if slot in self.scenario.conflict_slots or datetime.fromisoformat(slot).hour not in p["hours"] \
                or self._slot_taken(b["provider_id"], slot):
            return {"error": "slot_conflict"}
        b["slot"] = slot
        result = {"booking_id": booking_id, "status": "confirmed", "slot": slot}
        self.idem[idempotency_key] = result
        self._save()
        return dict(result)

    def list_bookings(self, user_id: str) -> dict:
        self._log("list_bookings", {"user_id": user_id})
        out = []
        for b in self.bookings.values():
            if b["user_id"] == user_id:
                out.append({"booking_id": b["booking_id"], "provider_id": b["provider_id"],
                            "provider_name": self.catalog[b["provider_id"]]["name"],
                            "service": b["service"], "slot": b["slot"], "status": b["status"],
                            "price": b["price"]})
        return {"bookings": sorted(out, key=lambda x: x["slot"])}

    # ----------------------------------------------------- simulation hooks (not agent-visible)
    def emit_provider_cancel(self, booking_id: str) -> None:
        """Simulated external event: the provider cancels a confirmed booking."""
        self.bookings[booking_id]["status"] = "cancelled_by_provider"
        self._save()

    def provider_hours(self, provider_id: str) -> list[int]:
        return list(self.catalog[provider_id]["hours"])

    def count_confirmed(self, provider_id: str | None = None, slot: str | None = None) -> int:
        return sum(1 for b in self.bookings.values() if b["status"] == "confirmed"
                   and (provider_id is None or b["provider_id"] == provider_id)
                   and (slot is None or b["slot"] == slot))
