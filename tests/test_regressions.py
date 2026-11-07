"""Regression tests for real bugs hit while building (see BUGS.md)."""
from agent.db import DB
from agent.llm import ScriptedLLM
from agent.orchestrator import CLAIM_RX
from agent.trace import sanitize
from evals.harness import booking_steps, make_agent
from mocks.tools import MockWorld, Scenario


def test_bug1_trace_sanitizer_keeps_iso_dates():
    out = sanitize({"slot": "2026-11-10T09:00", "date_range": {"start": "2026-11-09"}, "note": "call +91-9812345678"})
    assert out["slot"] == "2026-11-10T09:00" and out["date_range"]["start"] == "2026-11-09"
    assert "[redacted-phone]" in out["note"]


def test_bug2_claim_guard_ignores_negations_but_catches_claims():
    for safe in ["Nothing has been booked or changed.", "It has not been booked.", "I haven't booked anything."]:
        assert not CLAIM_RX.search(safe), safe
    for claim in ["I've booked it for you", "Your slot has been booked", "We have rescheduled it", "You're all set"]:
        assert CLAIM_RX.search(claim), claim


class Crash(BaseException):
    pass


def test_bug3_recovery_notification_on_restart_does_not_crash(tmp_path):
    def crash(_):
        raise Crash()

    world = MockWorld(Scenario(book_outcomes=["success"], after_book_commit=crash))
    agent, _, _ = make_agent(booking_steps(), world=world, db=DB(tmp_path / "a.db"))
    agent.handle("plumber this week mornings under 1500")
    try:
        agent.handle("yes")
    except Crash:
        pass
    world.scenario.after_book_commit = None
    agent2, _, _ = make_agent(world=world, db=DB(tmp_path / "a.db"), llm=ScriptedLLM([]))   # used to TypeError
    assert agent2.drain_outbox()
