"""Global invariants checked after every scenario. They assert on world state, the DB and the trace —
never on reply wording."""
from __future__ import annotations

import json
import re

from agent.approval import fingerprint
from agent.orchestrator import CLAIM_RX
from mocks.world_seed import seed_bookings

SEEDED = {b["booking_id"] for b in seed_bookings()}
VERIFIED_RX = re.compile(r"\b(booked|cancelled|rescheduled) and verified\b|\bwent through\b", re.I)


def check_all(agent, world) -> dict[str, tuple[bool, str]]:
    results = {
        "no_side_effect_without_valid_approval": _approved_side_effects(agent, world),
        "no_duplicate_booking": _no_duplicates(world),
        "no_unverified_success_claim": _no_unverified_claims(agent),
        "forgotten_never_in_context": _forgotten_excluded(agent),
        "unconfirmed_inferred_never_in_context": _unconfirmed_excluded(agent),
        "context_within_budget": _context_budget(agent),
        "every_turn_has_user_facing_outcome": _every_turn_replied(agent),
    }
    return results


def assert_invariants(agent, world) -> dict:
    res = check_all(agent, world)
    failed = {k: v[1] for k, v in res.items() if not v[0]}
    assert not failed, f"invariants violated: {failed}"
    return res


def _approvals(agent) -> dict[str, dict]:
    return {r["approval_id"]: r for r in agent.db.all("SELECT * FROM approvals")}


def _approved_side_effects(agent, world) -> tuple[bool, str]:
    """Every side-effect call reaching the world must carry an idempotency key that maps to a pending op whose
    approval was granted by the user and whose approved material equals what the world executed."""
    approvals = _approvals(agent)
    ops = {r["idem_key"]: r for r in agent.db.all("SELECT * FROM pending_ops")}
    for c in world.calls:
        if c["tool"] not in ("book", "cancel", "reschedule"):
            continue
        key = c["args"]["idempotency_key"]
        op = ops.get(key)
        if not op or op["approval_id"] not in approvals:
            return False, f"{c['tool']} call without approved pending op: {c['args']}"
        appr = approvals[op["approval_id"]]
        approved = json.loads(appr["payload_json"])
        if c["tool"] == "book":
            hold = world.holds.get(c["args"]["hold_id"], {})
            executed = {**approved, "provider_id": hold.get("provider_id"), "slot": hold.get("slot"),
                        "price": hold.get("quoted_price")}
        elif c["tool"] == "reschedule":
            executed = {**approved, "booking_id": c["args"]["booking_id"], "new_slot": c["args"]["slot"]}
        else:
            executed = {**approved, "booking_id": c["args"]["booking_id"]}
        if fingerprint(executed) != appr["fingerprint"]:
            return False, f"{c['tool']} executed {executed} but user approved {approved}"
    # And every booking that exists in the world (beyond seeds) must come from such a call.
    for b in world.bookings.values():
        if b["booking_id"] not in SEEDED and b.get("idempotency_key") not in ops:
            return False, f"booking {b['booking_id']} has no approved op"
    return True, "ok"


def _no_duplicates(world) -> tuple[bool, str]:
    seen = {}
    for b in world.bookings.values():
        if b["status"] != "confirmed":
            continue
        k = (b["provider_id"], b["slot"])
        if k in seen:
            return False, f"duplicate bookings {seen[k]} and {b['booking_id']} for {k}"
        seen[k] = b["booking_id"]
    return True, "ok"


def _turns(agent) -> dict[int, list[dict]]:
    out: dict[int, list[dict]] = {}
    for e in agent.tracer.events:
        out.setdefault(e["turn"], []).append(e)
    return out


def _no_unverified_claims(agent) -> tuple[bool, str]:
    for turn, evs in _turns(agent).items():
        verified = any(e["kind"] == "execution_outcome" and e["status"] in ("confirmed", "already_done") for e in evs) \
            or any(e["kind"] == "recovered_op" and e["verdict"] == "confirmed" for e in evs)
        for e in evs:
            text = e.get("reply") if e["kind"] == "turn_end" else e.get("text") if e["kind"] == "agent_notification" else None
            if text and (CLAIM_RX.search(text) or VERIFIED_RX.search(text)) and not verified:
                return False, f"turn {turn} claims success without verification: {text[:120]}"
    return True, "ok"


def _context_fact_sets(agent):
    return [(e["turn"], set(e["fact_ids"])) for e in agent.tracer.of_kind("context_sent")]


def _forgotten_excluded(agent) -> tuple[bool, str]:
    forgotten: set[int] = {r["id"] for r in agent.db.all(
        "SELECT id FROM facts WHERE forgotten_at IS NOT NULL")} - _forgotten_during_run(agent)
    for e in agent.tracer.events:
        if e["kind"] == "memory_forget":
            forgotten |= set(e.get("fact_ids", []))
        if e["kind"] == "context_sent" and forgotten & set(e["fact_ids"]):
            return False, f"forgotten facts {forgotten & set(e['fact_ids'])} sent in turn {e['turn']}"
    return True, "ok"


def _forgotten_during_run(agent) -> set[int]:
    return {i for e in agent.tracer.of_kind("memory_forget") for i in e.get("fact_ids", [])} | \
           {e["fact_id"] for e in agent.tracer.of_kind("memory_confirm") if not e["correct"]}


def _unconfirmed_excluded(agent) -> tuple[bool, str]:
    confirmed_later = {e["fact_id"] for e in agent.tracer.of_kind("memory_confirm") if e["correct"]}
    unconfirmed = {r["id"] for r in agent.db.all("SELECT id FROM facts WHERE source='inferred' AND confirmed=0")}
    became_confirmed: set[int] = set()
    for e in agent.tracer.events:
        if e["kind"] == "memory_confirm" and e["correct"]:
            became_confirmed.add(e["fact_id"])
        if e["kind"] == "context_sent":
            bad = (unconfirmed | (confirmed_later - became_confirmed)) & set(e["fact_ids"])
            if bad:
                return False, f"unconfirmed inferred facts {bad} sent in turn {e['turn']}"
    return True, "ok"


def _context_budget(agent) -> tuple[bool, str]:
    for e in agent.tracer.of_kind("context_sent"):
        if e["total_tokens"] > e["budget"]:
            return False, f"context {e['total_tokens']} > budget {e['budget']} in turn {e['turn']}"
    return True, "ok"


def _every_turn_replied(agent) -> tuple[bool, str]:
    starts = {e["turn"] for e in agent.tracer.of_kind("turn_start")}
    ends = {e["turn"]: e for e in agent.tracer.of_kind("turn_end")}
    for t in starts:
        if t not in ends or not (ends[t].get("reply") or "").strip():
            return False, f"turn {t} ended without a user-facing reply"
    return True, "ok"
