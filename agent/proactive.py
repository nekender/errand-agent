"""Proactive follow-up: detect provider-side cancellations without a user prompt.

Detection uses only the contract tools: poll list_bookings and diff against the agent's last-known
snapshot. For each newly cancelled-by-provider booking we decide whether to notify (only upcoming
bookings matter), search for a replacement within known preferences, hold the best one, and present it
for *fresh* approval. Nothing is ever rebooked automatically.
"""
from __future__ import annotations

from datetime import date, datetime, timedelta

from agent.booking import fmt_slot
from agent.state import Phase
from mocks.tools import ToolError


def tick(agent) -> list[str]:
    """Run one proactive check. Returns the messages it sent to the user (also queued in the outbox)."""
    tr, sent = agent.tracer, []
    try:
        bookings = agent.toolbox.call("list_bookings", user_id=agent.user_id)["bookings"]
    except ToolError as e:
        tr.emit("proactive_error", error=str(e))
        return []
    snaps = {r["booking_id"]: r for r in agent.db.all("SELECT * FROM booking_snapshots WHERE user_id=?",
                                                       (agent.user_id,))}
    for b in bookings:
        prev = snaps.get(b["booking_id"])
        agent.db.execute("INSERT INTO booking_snapshots(booking_id,user_id,status) VALUES(?,?,?) "
                         "ON CONFLICT(booking_id) DO UPDATE SET status=excluded.status",
                         (b["booking_id"], agent.user_id, b["status"]))
        if b["status"] != "cancelled_by_provider" or (prev and prev.get("handled_event")):
            continue
        agent.db.execute("UPDATE booking_snapshots SET handled_event='provider_cancel' WHERE booking_id=?",
                         (b["booking_id"],))
        tr.emit("proactive_event", event="provider_cancelled", booking=b)
        msg = _handle_provider_cancel(agent, b)
        if msg:
            agent.notify(msg, channel="proactive")
            sent.append(msg)
    agent.store.save(agent.session)
    return sent


def _handle_provider_cancel(agent, b: dict) -> str | None:
    tr, s, mem = agent.tracer, agent.session, agent.memory
    slot = datetime.fromisoformat(b["slot"])
    if slot <= agent.now():
        tr.emit("proactive_decision", decision="ignore", reason="booking already in the past")
        return None
    if s.phase in (Phase.EXECUTING, Phase.VERIFYING):
        tr.emit("proactive_decision", decision="notify_only", reason=f"session busy ({s.phase.value})")
        return f"Heads up: {b['provider_name']} cancelled your {b['service']} booking on {fmt_slot(b['slot'])}."

    # Replacement constraints come only from eligible memory + the original booking.
    area = mem.get("preferred_area") or agent.world.catalog[b["provider_id"]]["area"]
    earliest = int(mem.get("earliest_hour") or 0)
    budget = mem.get("budget_max", b["service"])
    budget = int(budget) if budget else int(b["price"] * 1.2)
    start = max(slot.date(), agent.now().date())
    rng = {"start": start.isoformat(), "end": (start + timedelta(days=3)).isoformat()}
    tr.emit("proactive_decision", decision="find_replacement",
            constraints={"area": area, "earliest_hour": earliest, "budget_max": budget, "date_range": rng})
    try:
        res = agent.toolbox.call("search_providers", service=b["service"], area=area, date_range=rng)
    except ToolError as e:
        tr.emit("proactive_error", error=str(e))
        return (f"{b['provider_name']} cancelled your {b['service']} booking on {fmt_slot(b['slot'])}. "
                "I couldn't search for a replacement right now — want me to try again?")

    def closeness(x: str) -> float:
        return abs((datetime.fromisoformat(x) - slot).total_seconds())

    options = []
    for p in res.get("providers", []):
        if p["provider_id"] == b["provider_id"] or p["price"] > budget:
            continue
        ok = [x for x in p["available_slots"] if datetime.fromisoformat(x).hour >= earliest]
        if ok:
            best = min(ok, key=closeness)
            options.append((closeness(best), -p["rating"], p, best))
    head = f"{b['provider_name']} cancelled your {b['service']} booking on {fmt_slot(b['slot'])}."
    if not options:
        tr.emit("proactive_decision", decision="no_replacement")
        return head + " I couldn't find a replacement within your preferences — want me to widen the search?"
    options.sort(key=lambda t: (t[0], t[1]))
    _, _, p, best = options[0]
    hold = agent.toolbox.call("hold_slot", provider_id=p["provider_id"], slot=best)
    if "error" in hold:
        return head + " The best replacement slot was just taken — want me to look again?"

    # Fresh request context for the replacement, then present it for approval (code-rendered).
    s.slots.merge({"service": b["service"], "area": area, "date_from": rng["start"], "date_to": rng["end"],
                   "earliest_hour": earliest, "budget_max": budget})
    s.candidates = [{"provider_id": p["provider_id"], "name": p["name"], "area": p["area"], "price": p["price"],
                     "rating": p["rating"], "attributes": p["attributes"], "within_budget": True, "preferred": False,
                     "matching_slots": [best], "all_slots": p["available_slots"]}]
    proposal = {"action": "book", "provider_id": p["provider_id"], "provider_name": p["name"], "service": b["service"],
                "slot": best, "price": hold["quoted_price"], "listed_price": p["price"], "hold_id": hold["hold_id"],
                "hold_expires_at": hold["expires_at"], "booking_id": None, "new_slot": None,
                "warnings": [], "replacement_for": b["booking_id"]}
    if hold["quoted_price"] != b["price"]:
        diff = hold["quoted_price"] - b["price"]
        proposal["warnings"].append(f"₹{abs(diff):,} {'more' if diff > 0 else 'less'} than the cancelled booking "
                                    f"(₹{b['price']:,})")
    agent.approvals.invalidate_session(s.session_id, "superseded_by_replacement")
    proposal["shown_turn"] = s.turn
    s.proposal = proposal
    agent.store.transition(s, Phase.AWAITING_APPROVAL, "proactive replacement proposal")
    tr.emit("proposal_shown", proposal={k: proposal[k] for k in ("action", "provider_id", "slot", "price", "hold_id")},
            proactive=True)
    return head + f" I found a replacement within your preferences ({p['rating']}★). " + agent._summary(proposal)
