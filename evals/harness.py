"""Helpers for building agents and scripted-model steps in eval scenarios."""
from __future__ import annotations

import json

from agent.db import DB
from agent.limits import Limits
from agent.llm import ScriptedLLM, call
from agent.orchestrator import Agent
from agent.seed import seed_conversation, seed_memories
from mocks.tools import MockWorld, Scenario


def make_agent(steps=None, scenario: Scenario | None = None, world: MockWorld | None = None, db: DB | None = None,
               seed_mem: bool = True, seed_convo: bool = False, limits: Limits | None = None,
               context_budget: int = 3000, session_id: str = "eval", default=None, llm=None):
    # Default: book() succeeds, so a scenario only exercises the failure modes it names explicitly.
    world = world or MockWorld(scenario or Scenario(book_outcomes=["success"] * 10))
    db = db or DB()
    fresh_db = db.one("SELECT COUNT(*) AS n FROM facts")["n"] == 0
    llm = llm or ScriptedLLM(steps or [], default=default)
    if fresh_db:
        from agent.memory import Memory
        if seed_mem:
            seed_memories(Memory(db, "u1", now_fn=world.clock.now))
        if seed_convo:
            seed_conversation(db, "u1")
    agent = Agent(world, llm, db=db, session_id=session_id, limits=limits or Limits(), trace_dir=None,
                  context_budget=context_budget)
    return agent, world, llm


def last_tool_result(messages: list[dict], key: str | None = None) -> dict:
    """The JSON of the most recent tool_result the scripted model received (optionally: containing `key`)."""
    for m in reversed(messages):
        if m["role"] == "user" and isinstance(m["content"], list):
            for b in m["content"]:
                if b.get("type") == "tool_result":
                    out = json.loads(b["content"])
                    if key is None or key in out:
                        return out
    return {}


def tool_errors(llm) -> list[str]:
    """Every error string the application returned to the scripted model."""
    errs = []
    for r in llm.requests:
        for m in r["messages"]:
            if m["role"] == "user" and isinstance(m["content"], list):
                for b in m["content"]:
                    if b.get("type") == "tool_result" and b.get("is_error"):
                        errs.append(json.loads(b["content"]).get("error", ""))
    return sorted(set(errs), key=errs.index)


def propose_top(min_hour: int | None = None, provider_id: str | None = None, index: int = 0):
    """Scripted step: propose the best candidate's first matching slot (as a careful model would)."""
    def step(system, messages, tools):
        res = last_tool_result(messages, "providers")
        provs = res["providers"]
        p = next(x for x in provs if x["provider_id"] == provider_id) if provider_id else \
            [x for x in provs if x["matching_slots"] and x["within_budget"]][index]
        slots = [s for s in p["matching_slots"] if min_hour is None or int(s[11:13]) >= min_hour]
        return call("propose_booking", provider_id=p["provider_id"], slot=slots[0])
    return step


def gather_plumber(**over):
    args = dict(service="plumber", date_from="2026-11-09", date_to="2026-11-15", latest_hour=11, budget_max=1500)
    args.update(over)
    return call("update_request", **args)


def booking_steps(**over):
    """Standard 'find a plumber this week, mornings, under ₹1500' model behaviour up to a proposal."""
    return [gather_plumber(**over), call("search_providers"), propose_top()]


def new_bookings(world) -> list[dict]:
    from evals.invariants import SEEDED
    return [b for b in world.bookings.values() if b["booking_id"] not in SEEDED]


def side_effect_calls(world, tool="book") -> list[dict]:
    return [c for c in world.calls if c["tool"] == tool]
