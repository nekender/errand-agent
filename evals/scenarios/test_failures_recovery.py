"""Booking correctness under tool failures, timeouts, conflicts, repeated confirmations and restarts."""
import pytest

from agent.db import DB
from agent.llm import ScriptedLLM, call, say
from agent.state import Phase
from evals.harness import booking_steps, make_agent, new_bookings, propose_top, side_effect_calls
from mocks.tools import MockWorld, Scenario


@pytest.mark.scenario("Booking timeout after success (booking was committed)",
                      expected="query get_booking_status before anything else; report verified success; exactly 1 booking; no retry")
def test_timeout_after_success(ev):
    agent, world, _ = make_agent(booking_steps(), Scenario(book_outcomes=["timeout_booked"]))
    agent.handle("plumber this week mornings under 1500")
    reply = agent.handle("yes")
    books = side_effect_calls(world, "book")
    kinds = [e["kind"] for e in agent.tracer.events if e["turn"] == 2]
    assert len(books) == 1 and len(new_bookings(world)) == 1
    assert kinds.index("ambiguous_outcome") < kinds.index("reconciled") < kinds.index("execution_outcome")
    assert any(e["kind"] == "reconciled" and e["verdict"] == "booked_despite_timeout" for e in agent.tracer.events)
    assert "verified" in reply and agent.session.phase == Phase.DONE
    ev.actual("timeout → get_booking_status=confirmed → reported verified; 1 book call, 1 booking")
    ev.invariants(agent, world)


@pytest.mark.scenario("Booking timeout, nothing committed",
                      expected="status=not_found → retry with the SAME idempotency key → exactly 1 booking")
def test_timeout_not_booked_retries_same_key(ev):
    agent, world, _ = make_agent(booking_steps(), Scenario(book_outcomes=["timeout_not_booked", "success"]))
    agent.handle("plumber this week mornings under 1500")
    agent.handle("yes")
    books = side_effect_calls(world, "book")
    keys = {c["args"]["idempotency_key"] for c in books}
    assert len(books) == 2 and len(keys) == 1 and len(new_bookings(world)) == 1
    ev.actual(f"2 book calls with one key {next(iter(keys))[:14]}…; 1 booking")
    ev.invariants(agent, world)


@pytest.mark.scenario("Booking keeps timing out (retries exhausted)",
                      expected="bounded retries (3), verified nothing booked, clear failure message, no hang")
def test_timeout_retries_exhausted(ev):
    agent, world, _ = make_agent(booking_steps(), Scenario(book_outcomes=["timeout_not_booked"] * 5))
    agent.handle("plumber this week mornings under 1500")
    reply = agent.handle("yes")
    assert len(side_effect_calls(world, "book")) == 3 and not new_bookings(world)
    assert agent.session.phase == Phase.FAILED and "nothing was booked" in reply.lower()
    ev.actual("3 attempts, each verified not booked → FAILED with explanation")
    ev.invariants(agent, world)


@pytest.mark.scenario("Slot taken at booking time",
                      expected="no booking, alternatives offered, approval consumed; new proposal needs a new yes")
def test_slot_taken(ev):
    agent, world, _ = make_agent(booking_steps() + [call("search_providers"), propose_top(index=1)],
                                 Scenario(book_outcomes=["slot_taken", "success"]))
    agent.handle("plumber this week mornings under 1500")
    r = agent.handle("yes")
    assert "taken" in r and "Alternatives" in r and not new_bookings(world)
    assert agent.session.phase == Phase.COMPARING
    agent.handle("ok propose the next one")           # not an approval: goes to the model, which proposes
    assert agent.session.phase == Phase.AWAITING_APPROVAL and not new_bookings(world)
    agent.handle("yes")
    assert len(new_bookings(world)) == 1
    ev.actual("slot_taken → alternatives, COMPARING; second proposal booked only after a second yes")
    ev.invariants(agent, world)


@pytest.mark.scenario("Hold expired at provider (clock skew)",
                      expected="book returns hold_expired → re-hold same slot at same price → booked under same key")
def test_hold_expired_server_side(ev):
    agent, world, _ = make_agent(booking_steps())
    agent.handle("plumber this week mornings under 1500")
    hold_id = agent.session.proposal["hold_id"]
    world.holds[hold_id]["expires_at"] = "2026-01-01T00:00:00"   # provider thinks it expired; we don't
    agent.handle("yes")
    holds = side_effect_calls(world, "hold_slot")
    assert len(holds) == 2 and len(new_bookings(world)) == 1
    assert len({c["args"]["idempotency_key"] for c in side_effect_calls(world, "book")}) == 1
    ev.actual("hold_expired → re-hold (price unchanged, approval still valid) → 1 booking")
    ev.invariants(agent, world)


@pytest.mark.scenario("Repeated confirmation",
                      expected="'yes' ×3 → exactly one booking, later yeses answered from state with no tool calls")
def test_repeated_confirmation(ev):
    agent, world, _ = make_agent(booking_steps())
    agent.handle("plumber this week mornings under 1500")
    agent.handle("yes")
    agent.handle("yes")
    r3 = agent.handle("Yes, book it")
    assert len(new_bookings(world)) == 1 and len(side_effect_calls(world, "book")) == 1
    assert "already" in r3
    ev.actual("1 book call, 1 booking; duplicate confirmations recognised")
    ev.invariants(agent, world)


class SimulatedCrash(BaseException):
    """Process death: not an Exception, so nothing in the agent can swallow it."""


@pytest.mark.scenario("Restart mid-booking, after the provider committed",
                      expected="on restart, in-flight op reconciled via get_booking_status; user notified; no duplicate")
def test_restart_after_commit(ev, tmp_path):
    db = DB(tmp_path / "agent.db")

    def crash(_):
        raise SimulatedCrash()

    world = MockWorld(Scenario(book_outcomes=["success"], after_book_commit=crash))
    agent, _, _ = make_agent(booking_steps(), world=world, db=db)
    agent.handle("plumber this week mornings under 1500")
    with pytest.raises(SimulatedCrash):
        agent.handle("yes")
    assert db.one("SELECT status FROM pending_ops")["status"] == "in_flight"
    world.scenario.after_book_commit = None
    # New process: same DB + same provider world, fresh model.
    agent2, _, _ = make_agent(world=world, db=DB(tmp_path / "agent.db"), llm=ScriptedLLM([]))
    notes = agent2.drain_outbox()
    assert notes and "went through" in notes[0]
    r = agent2.handle("yes")
    assert len(new_bookings(world)) == 1 and len(side_effect_calls(world, "book")) == 1 and "already" in r
    ev.actual("crash after commit → restart reconciled op → 'went through (verified)'; extra yes made no booking")
    ev.invariants(agent2, world)


@pytest.mark.scenario("Restart mid-booking, before the provider committed",
                      expected="restart verifies nothing booked → re-presents proposal; booking needs a fresh yes")
def test_restart_before_commit(ev, tmp_path):
    db = DB(tmp_path / "agent.db")
    world = MockWorld(Scenario(book_outcomes=["success"]))
    agent, _, _ = make_agent(booking_steps(), world=world, db=db)
    agent.handle("plumber this week mornings under 1500")
    real_book = world.book

    def dying_book(**kw):
        raise SimulatedCrash()

    world.book = dying_book
    with pytest.raises(SimulatedCrash):
        agent.handle("yes")
    world.book = real_book
    agent2, _, _ = make_agent(world=world, db=DB(tmp_path / "agent.db"), llm=ScriptedLLM([]))
    notes = agent2.drain_outbox()
    assert "did NOT go through" in notes[0] and agent2.session.phase == Phase.AWAITING_APPROVAL
    assert not new_bookings(world)
    agent2.handle("yes")
    assert len(new_bookings(world)) == 1
    ev.actual("restart → status not_found → proposal re-presented → booked once after fresh yes")
    ev.invariants(agent2, world)


@pytest.mark.scenario("Cancel with transient 503",
                      expected="first cancel 503 → retried with same key → verified cancelled")
def test_cancel_503_retry(ev):
    agent, world, _ = make_agent([call("list_bookings"), call("propose_cancel", booking_id="bk_up_ac01")])
    agent.handle("Cancel my AC servicing")
    assert world.bookings["bk_up_ac01"]["status"] == "confirmed"
    r = agent.handle("yes")
    cancels = side_effect_calls(world, "cancel")
    assert len(cancels) == 2 and len({c["args"]["idempotency_key"] for c in cancels}) == 1
    assert world.bookings["bk_up_ac01"]["status"] == "cancelled" and "verified" in r
    ev.actual("503 then success, same key; booking cancelled & verified")
    ev.invariants(agent, world)


@pytest.mark.scenario("Reschedule slot conflict",
                      expected="slot_conflict → booking unchanged, alternatives offered; second slot needs new yes")
def test_reschedule_conflict(ev):
    agent, world, _ = make_agent(
        [call("list_bookings"), call("propose_reschedule", booking_id="bk_up_ac01", new_slot="2026-11-14T10:00"),
         call("propose_reschedule", booking_id="bk_up_ac01", new_slot="2026-11-14T11:00")],
        Scenario(conflict_slots=["2026-11-14T10:00"]))
    agent.handle("Move my AC servicing to Saturday")
    r = agent.handle("yes")
    assert "conflicts" in r and world.bookings["bk_up_ac01"]["slot"] == "2026-11-12T10:00"
    agent.handle("try 11 instead")
    assert world.bookings["bk_up_ac01"]["slot"] == "2026-11-12T10:00"   # still unchanged before approval
    agent.handle("yes")
    assert world.bookings["bk_up_ac01"]["slot"] == "2026-11-14T11:00"
    ev.actual("conflict left booking intact + alternatives; 11:00 applied only after second yes")
    ev.invariants(agent, world)


@pytest.mark.scenario("Unexpected tool error, then recovery",
                      expected="search crash returned to model as error; model retries; flow completes")
def test_unexpected_tool_error(ev):
    agent, world, _ = make_agent([call("update_request", service="plumber", date_from="2026-11-09",
                                       date_to="2026-11-15", latest_hour=11, budget_max=1500),
                                  call("search_providers"), call("search_providers"), propose_top()],
                                 Scenario(book_outcomes=["success"], tool_crashes={"search_providers": 1}))
    agent.handle("plumber this week mornings under 1500")
    assert agent.session.phase == Phase.AWAITING_APPROVAL
    assert any(e["kind"] == "tool_handler_error" for e in agent.tracer.events)
    ev.actual("first search raised; error surfaced to model; retry succeeded; proposal shown")
    ev.invariants(agent, world)
