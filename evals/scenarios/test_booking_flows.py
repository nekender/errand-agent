"""Core booking / clarification / reschedule flows driven by a simulated user."""
import pytest

from agent.llm import call, say
from agent.state import Phase
from evals.harness import tool_errors, booking_steps, last_tool_result, make_agent, new_bookings, propose_top, side_effect_calls
from evals.sim_user import SimUser, approve_if


@pytest.mark.scenario("Happy path: plumber this week, mornings, under ₹1,500",
                      expected="one verified booking matching the approved proposal; slot ≤11:00; price ≤1500")
def test_happy_path(ev):
    agent, world, _ = make_agent(booking_steps())
    user = SimUser("Find a plumber for my kitchen tap this week, mornings only, under ₹1,500.",
                   rules=[(r"Please confirm", approve_if(max_price=1500))])
    user.run(agent)
    bks = new_bookings(world)
    assert len(bks) == 1 and agent.session.phase == Phase.DONE
    b = bks[0]
    assert int(b["slot"][11:13]) <= 11 and b["price"] <= 1500
    ev.actual(f"booked {b['provider_id']} {b['slot']} ₹{b['price']}; phase DONE; {len(user.transcript)} turns")
    ev.invariants(agent, world)


@pytest.mark.scenario("Clarify only missing details; never re-ask supplied facts",
                      expected="area & earliest hour filled from memory; only dates+budget missing; service kept")
def test_clarify_missing_only(ev):
    agent, world, llm = make_agent([
        call("update_request", service="electrician"),
        say("Which dates work for you, and what's your budget?"),
        call("update_request", date_from="2026-11-10", date_to="2026-11-12"),
        say("And your budget?"),
    ])
    agent.handle("I need an electrician to fix a switchboard")
    first = agent.session.slots.missing()
    agent.handle("Tuesday to Thursday works")
    second = agent.session.slots.missing()
    sl = agent.session.slots
    assert first == ["date_from", "date_to", "budget_max"], first
    assert second == ["budget_max"], second
    assert sl.service == "electrician" and sl.area == "Koramangala" and sl.earliest_hour == 9
    assert set(sl.from_memory) >= {"area", "earliest_hour"}
    # The model is told exactly what is missing — the context it received on turn 2 lists only budget/dates.
    assert '"missing": ["date_from", "date_to", "budget_max"]' in llm.requests[2]["messages"][-1]["content"]
    ev.actual(f"missing after t1={first}, after t2={second}; area/earliest from memory")
    ev.invariants(agent, world)


@pytest.mark.scenario("Correction preserves unrelated context",
                      expected="changing budget keeps service/area/dates/candidates; changing area clears candidates only")
def test_correction_preserves_context(ev):
    agent, world, _ = make_agent([
        call("update_request", service="plumber", date_from="2026-11-09", date_to="2026-11-15", budget_max=1500),
        call("search_providers"), say("Here are options."),
        call("update_request", budget_max=1200), say("Updated budget."),
        call("update_request", area="Indiranagar"), say("Updated area; I'll search again."),
    ])
    agent.handle("plumber this week under 1500")
    before = agent.session.slots.to_dict()
    n_cands = len(agent.session.candidates)
    agent.handle("actually make the budget 1200")
    mid = agent.session.slots.to_dict()
    assert {k: v for k, v in mid.items() if k != "budget_max"} == {k: v for k, v in before.items() if k != "budget_max"}
    assert mid["budget_max"] == 1200 and len(agent.session.candidates) == n_cands
    agent.handle("oh and it's at my Indiranagar flat")
    after = agent.session.slots.to_dict()
    assert after["area"] == "Indiranagar" and after["service"] == "plumber" and after["date_to"] == "2026-11-15"
    assert after["budget_max"] == 1200 and agent.session.candidates == []
    ev.actual("budget-only change kept all other fields & candidates; area change kept other fields, cleared stale results")
    ev.invariants(agent, world)


@pytest.mark.scenario("Move my AC servicing to Saturday",
                      expected="reschedule proposal for bk_up_ac01 → Sat 10:00; executed only after yes; verified")
def test_reschedule_to_saturday(ev):
    agent, world, _ = make_agent([
        call("list_bookings"),
        call("propose_reschedule", booking_id="bk_up_ac01", new_slot="2026-11-14T10:00"),
    ])
    r1 = agent.handle("Move my AC servicing to Saturday.")
    assert agent.session.phase == Phase.AWAITING_APPROVAL and not side_effect_calls(world, "reschedule")
    r2 = agent.handle("yes")
    assert world.bookings["bk_up_ac01"]["slot"] == "2026-11-14T10:00"
    assert len(side_effect_calls(world, "reschedule")) == 1 and "verified" in r2
    ev.actual("proposal shown, no side effect until 'yes'; booking moved to 2026-11-14T10:00 and verified")
    ev.invariants(agent, world)


@pytest.mark.scenario("Same electrician as last time, not before 10 a.m.",
                      expected="uses elec_02 from history; code rejects a 9:00 slot; books ≥10:00")
def test_same_electrician_not_before_10(ev):
    def propose_9am(system, messages, tools):
        return call("propose_booking", provider_id="elec_02", slot="2026-11-10T09:00")

    agent, world, llm = make_agent([
        call("list_bookings"),
        call("update_request", service="electrician", provider_id="elec_02", earliest_hour=10,
             date_from="2026-11-10", date_to="2026-11-13", budget_max=1200),
        call("search_providers"),
        propose_9am,                       # a sloppy model picks 9:00 — must be rejected by code
        propose_top(provider_id="elec_02", min_hour=10),
    ])
    agent.handle("Book the same electrician as last time, but not before 10 a.m. Budget 1200.")
    rejected = [e for e in tool_errors(llm) if "violates the user's date/time constraints" in e]
    assert rejected, "9:00 slot should have been rejected by application code"
    assert agent.session.proposal["provider_id"] == "elec_02"
    agent.handle("yes")
    b = new_bookings(world)[0]
    assert b["provider_id"] == "elec_02" and int(b["slot"][11:13]) >= 10
    ev.actual(f"9:00 proposal rejected by code; booked elec_02 at {b['slot']}")
    ev.invariants(agent, world)
