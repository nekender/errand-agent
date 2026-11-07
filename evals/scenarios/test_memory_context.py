"""Persistent memory, context budget, limits and proactive follow-up."""
import pytest

from agent import proactive
from agent.limits import Limits
from agent.llm import call, say
from agent.state import Phase
from evals.harness import booking_steps, make_agent, new_bookings, side_effect_calls


def ctx_events(agent):
    return agent.tracer.of_kind("context_sent")


def fact_id(agent, key, scope="global", value=None):
    rows = agent.db.all("SELECT id, value FROM facts WHERE key=? AND scope=?", (key, scope))
    return next(r["id"] for r in rows if value is None or r["value"] == value)


@pytest.mark.scenario("Memory: inspect what is stored",
                      expected="explicit vs inferred (confirmed/unconfirmed) grouped; forgotten facts absent")
def test_memory_inspect(ev):
    agent, world, llm = make_agent([call("list_memory"), say("Here's what I remember.")])
    agent.handle("What do you remember about me?")
    inv = agent.memory.inspect()
    keys = {f["key"] for group in inv.values() for f in group}
    assert "preferred_area" in {f["key"] for f in inv["explicit"]}
    assert {"prefers_weekends", "latest_hour"} <= {f["key"] for f in inv["inferred_unconfirmed"]}
    assert "language" in {f["key"] for f in inv["inferred_confirmed"]}
    assert "old_area" not in keys
    ev.actual(f"{len(inv['explicit'])} explicit, {len(inv['inferred_confirmed'])} inferred-confirmed, "
              f"{len(inv['inferred_unconfirmed'])} inferred-unconfirmed; forgotten hidden")
    ev.invariants(agent, world)


@pytest.mark.scenario("Memory: correct a stored fact",
                      expected="new value replaces old (same fact id); next turn's context has only the new value; "
                               "new requests default to it")
def test_memory_correct(ev):
    agent, world, _ = make_agent([
        call("remember_preference", key="preferred_area", value="Indiranagar"), say("Updated."),
        call("update_request", service="plumber"), say("When?"),
    ])
    fid = fact_id(agent, "preferred_area")
    agent.handle("I've moved — my area is Indiranagar now")
    agent.handle("I need a plumber")
    assert agent.memory.get("preferred_area") == "Indiranagar" and fact_id(agent, "preferred_area") == fid
    assert agent.session.slots.area == "Indiranagar"
    ev.actual("fact updated in place; request area defaulted to Indiranagar")
    ev.invariants(agent, world)


@pytest.mark.scenario("Memory: forget a fact",
                      expected="forgotten fact never sent to the model again and no longer drives ranking")
def test_memory_forget(ev):
    agent, world, _ = make_agent([
        call("update_request", service="electrician"), say("When?"),
        call("forget_preference", key="preferred_provider", scope="electrician"), say("Forgotten."),
        call("update_request", date_from="2026-11-10", date_to="2026-11-12"), say("Budget?"),
    ])
    fid = fact_id(agent, "preferred_provider", "electrician")
    agent.handle("I need an electrician")
    agent.handle("forget my preferred electrician")
    # Context is built at the start of each turn: turn 2 (service now known) still carried the fact.
    assert fid in ctx_events(agent)[-1]["fact_ids"]
    agent.handle("Tuesday to Thursday")
    assert fid not in ctx_events(agent)[-1]["fact_ids"] and agent._preferred_provider() is None
    assert fid not in [f["id"] for f in agent.memory.eligible()]
    ev.actual(f"fact {fid} in context before forget, absent after; preference no longer applied")
    ev.invariants(agent, world)


@pytest.mark.scenario("Unconfirmed inferred facts stay out of context until confirmed",
                      expected="inferred 'prefers_weekends' never sent; after user confirms it is eligible")
def test_inferred_needs_confirmation(ev):
    agent, world, _ = make_agent([say("Hi!"), call("confirm_inferred", fact_id=11, correct=True), say("Noted."),
                                  say("ok")])
    fid = fact_id(agent, "prefers_weekends")
    unconfirmed = [r["id"] for r in agent.db.all("SELECT id FROM facts WHERE source='inferred' AND confirmed=0")]
    agent.handle("hello")
    assert not set(unconfirmed) & set(ctx_events(agent)[-1]["fact_ids"])
    assert agent.session.slots.latest_hour is None          # inferred latest_hour=18 not applied as a default
    assert fid == 11
    agent.handle("yes, I do prefer weekends")
    agent.handle("thanks")
    assert fid in ctx_events(agent)[-1]["fact_ids"]
    ev.actual(f"{len(unconfirmed)} unconfirmed inferred facts never sent; fact {fid} sent only after confirmation")
    ev.invariants(agent, world)


@pytest.mark.scenario("Context budget with 50+ memories and a 200-message history",
                      expected="every model context ≤ budget; old history omitted; other-service facts excluded; log proves it")
def test_context_budget(ev):
    agent, world, _ = make_agent(booking_steps(), seed_convo=True, context_budget=2200)
    n_facts = agent.db.one("SELECT COUNT(*) AS n FROM facts")["n"]
    n_msgs = agent.db.one("SELECT COUNT(*) AS n FROM messages")["n"]
    agent.handle("plumber this week mornings under 1500")
    e = ctx_events(agent)[-1]
    sent = set(e["fact_ids"])
    cleaning = fact_id(agent, "cleaning_notes", "cleaning")
    ac = fact_id(agent, "ac_details", "ac_service")
    assert n_facts >= 50 and n_msgs >= 200
    assert e["total_tokens"] <= e["budget"] and e["omitted_messages"] > 0
    assert cleaning not in sent and ac not in sent
    row = agent.db.one("SELECT * FROM context_log ORDER BY id DESC LIMIT 1")
    assert row["total_tokens"] <= row["budget"] == 2200
    ev.actual(f"{n_facts} facts/{n_msgs} msgs stored → sent {len(sent)} facts, {e['message_ids_count']} msgs, "
              f"{e['total_tokens']}/{e['budget']} tokens; {e['omitted_messages']} msgs omitted")
    ev.invariants(agent, world)


@pytest.mark.scenario("Limit exhaustion: model loops forever",
                      expected="stops at max_model_calls with a clear message; nothing booked; no hang")
def test_model_call_limit(ev):
    agent, world, _ = make_agent(default=call("list_bookings"), limits=Limits(max_model_calls=5, max_tool_calls=50))
    r = agent.handle("what bookings do I have?")
    assert "limit" in r and len(agent.tracer.of_kind("model_call")) == 5
    assert agent.tracer.of_kind("turn_end")[-1]["outcome"] == "limit:max_model_calls"
    ev.actual("halted after 5 model calls with user-facing limit message")
    ev.invariants(agent, world)


@pytest.mark.scenario("Limit exhaustion: tool-call storm and repeated tool failures",
                      expected="max_tool_calls and consecutive-error limits each end the turn cleanly")
def test_tool_limits(ev):
    storm = {"tool_calls": [{"name": "list_bookings", "args": {}}] * 30}
    agent, world, _ = make_agent([storm], limits=Limits(max_tool_calls=10))
    r1 = agent.handle("list everything")
    from mocks.tools import Scenario
    agent2, world2, _ = make_agent(default=call("search_providers"),
                                   scenario=Scenario(tool_crashes={"search_providers": 99}))
    agent2.session.slots.merge({"service": "plumber", "area": "Koramangala", "date_from": "2026-11-09",
                                "date_to": "2026-11-12"})
    r2 = agent2.handle("search please")
    assert agent.tracer.of_kind("turn_end")[-1]["outcome"] == "limit:max_tool_calls" and "limit" in r1
    assert agent2.tracer.of_kind("turn_end")[-1]["outcome"] == "limit:max_consecutive_tool_errors" and "limit" in r2
    ev.actual("30-call storm stopped at 10 tool calls; 4 consecutive tool errors stopped the turn; both replied")
    ev.invariants(agent, world)
    ev.invariants(agent2, world2)


@pytest.mark.scenario("Proactive: provider cancels an upcoming booking",
                      expected="detected without user prompt; replacement within prefs (area, ≥9am, budget) held and "
                               "proposed; nothing booked until fresh yes")
def test_proactive_cancellation(ev):
    agent, world, _ = make_agent([])
    proactive.tick(agent)                                  # baseline snapshot
    world.emit_provider_cancel("bk_up_ac01")
    sent = proactive.tick(agent)                           # no user message involved
    assert len(sent) == 1 and "cancelled" in sent[0] and "Please confirm" in sent[0]
    p = agent.session.proposal
    assert agent.session.phase == Phase.AWAITING_APPROVAL and p["replacement_for"] == "bk_up_ac01"
    assert p["provider_id"] != "ac_s_01" and int(p["slot"][11:13]) >= 9 and not side_effect_calls(world, "book")
    assert proactive.tick(agent) == []                     # handled once
    agent.handle("yes")
    b = new_bookings(world)
    assert len(b) == 1 and b[0]["provider_id"] == p["provider_id"]
    ev.actual(f"notified + proposed {p['provider_id']} {p['slot']} ₹{p['price']}; booked only after yes")
    ev.invariants(agent, world)


@pytest.mark.scenario("Proactive: provider cancels a past booking",
                      expected="decides not to notify (no action needed)")
def test_proactive_past_ignored(ev):
    agent, world, _ = make_agent([])
    world.emit_provider_cancel("bk_hist_001")
    assert proactive.tick(agent) == []
    assert any(e["kind"] == "proactive_decision" and e["decision"] == "ignore" for e in agent.tracer.events)
    ev.actual("past booking cancellation detected and deliberately ignored")
    ev.invariants(agent, world)
