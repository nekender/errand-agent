"""Demo UI: the agent's chat on the left, its live trace on the right.

    streamlit run ui/app.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")

from agent import proactive  # noqa: E402
from agent.db import DB  # noqa: E402
from agent.limits import Limits  # noqa: E402
from agent.memory import Memory  # noqa: E402
from agent.orchestrator import Agent  # noqa: E402
from agent.seed import seed_conversation, seed_memories  # noqa: E402
from mocks.tools import MockWorld, Scenario  # noqa: E402

st.set_page_config(page_title="Errand Agent", layout="wide")
STATE = ROOT / ".state"

KIND_STYLE = {
    "state_transition": "🔀", "model_call": "🧠", "tool_call": "🔧", "model_tool": "🛠️", "context_sent": "📦",
    "approval_granted": "✅", "approval_invalidated": "⛔", "approval_consumed": "🔒", "proposal_shown": "📝",
    "pending_op_written": "💾", "pending_op_closed": "💾", "ambiguous_outcome": "⚠️", "reconciled": "🔎",
    "injection_detected": "🛡️", "claim_blocked": "🛡️", "ungrounded_update_rejected": "🛡️",
    "duplicate_prevented": "♻️", "execution_outcome": "🏁", "turn_start": "▶️", "turn_end": "⏹️",
    "proactive_event": "📣", "proactive_decision": "📣", "agent_notification": "📣", "recovered_op": "🩹",
    "error": "❌", "slots_updated": "🧩", "memory_write": "🧠", "memory_forget": "🗑️",
}


def new_agent(fresh: bool) -> Agent:
    STATE.mkdir(exist_ok=True)
    if fresh:
        for f in STATE.glob("ui_*"):
            f.unlink()
    db = DB(STATE / "ui_agent.db")
    world = MockWorld(Scenario(seed=42), persist_path=STATE / "ui_world.json")
    if db.one("SELECT COUNT(*) AS n FROM facts")["n"] == 0:
        seed_memories(Memory(db, "u1", now_fn=world.clock.now))
        seed_conversation(db, "u1")
    if os.getenv("ERRAND_OFFLINE_DEMO"):
        # Offline smoke mode: scripted model running the standard plumber flow (no API key needed).
        from agent.llm import ScriptedLLM, say
        from evals.harness import booking_steps
        llm = ScriptedLLM(booking_steps(), default=say("(offline demo model: no more scripted steps)"))
    else:
        from agent.llm import AnthropicLLM
        llm = AnthropicLLM()
    agent = Agent(world, llm, db=db, session_id="ui", limits=Limits(retry_backoff_s=0.2),
                  trace_dir=ROOT / "traces")
    proactive.tick(agent)  # baseline snapshot of bookings
    return agent


if "agent" not in st.session_state:
    try:
        st.session_state.agent = new_agent(fresh=True)
        st.session_state.chat = []
    except Exception as e:  # e.g. missing credentials
        st.error(f"Could not start the agent: {e}. Put ANTHROPIC_API_KEY in .env and reload.")
        st.stop()

agent: Agent = st.session_state.agent
world: MockWorld = agent.world
chat: list = st.session_state.chat

# ------------------------------------------------------------------ sidebar: failure injection
with st.sidebar:
    st.header("Scenario controls")
    st.caption(f"Mock clock: {world.clock.now():%a %d %b %H:%M}")
    nxt = st.selectbox("Next book() outcome", ["random (70/15/15)", "success", "timeout_booked",
                                               "timeout_not_booked", "slot_taken"])
    if st.button("Apply outcome"):
        world.scenario.book_outcomes = [] if nxt.startswith("random") else [nxt]
        st.toast(f"next book(): {nxt}")
    delta = st.number_input("Hold price change (₹)", value=0, step=50)
    if st.button("Apply price change"):
        world.scenario.hold_price_delta = {"*": int(delta)} if delta else {}
    if st.button("Advance clock 11 min (expire holds)"):
        world.clock.advance(minutes=11)
    upcoming = [b for b in world.list_bookings("u1")["bookings"] if b["status"] == "confirmed"]
    if upcoming:
        bid = st.selectbox("Booking", [b["booking_id"] for b in upcoming],
                           format_func=lambda i: next(f"{i} · {b['service']} · {b['slot']}" for b in upcoming
                                                      if b["booking_id"] == i))
        if st.button("Simulate provider cancellation"):
            world.emit_provider_cancel(bid)
            st.toast("provider cancelled — agent will notice on its next check")
    st.divider()
    with st.expander("Memory"):
        st.json(agent.memory.inspect(), expanded=False)
    with st.expander("Bookings (source of truth)"):
        st.json(world.list_bookings("u1"), expanded=False)
    if st.button("Reset demo"):
        del st.session_state["agent"]
        st.rerun()

left, right = st.columns([1, 1], gap="large")

# --------------------------------------------------------------------- proactive check (no user input)
for m in agent.drain_outbox():
    chat.append(("assistant", m, "notice"))
for m in proactive.tick(agent):
    pass  # tick() queues its own messages in the outbox
for m in agent.drain_outbox():
    chat.append(("assistant", m, "proactive"))

with left:
    st.subheader("Errand agent")
    st.caption(f"phase: **{agent.session.phase.value}** · model: {agent.llm.name}")
    box = st.container(height=620)
    with box:
        for role, text, tag in chat:
            with st.chat_message(role, avatar="📣" if tag == "proactive" else None):
                st.markdown(text)
    prompt = st.chat_input("e.g. Find a plumber for my kitchen tap this week, mornings only, under ₹1,500")
    if prompt:
        chat.append(("user", prompt, ""))
        with st.spinner("thinking…"):
            reply = agent.handle(prompt)
        chat.append(("assistant", reply, ""))
        st.rerun()

with right:
    t = agent.tracer.totals()
    st.subheader("Trace")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("model calls", t["model_calls"])
    c2.metric("tool calls", t["tool_calls"])
    c3.metric("tokens in/out", f"{t['input_tokens']}/{t['output_tokens']}")
    c4.metric("est. cost", f"${t['cost_usd']:.4f}")
    only_turn = st.toggle("Latest turn only", value=True)
    trace_box = st.container(height=560)
    events = agent.tracer.events
    if only_turn and events:
        events = [e for e in events if e["turn"] == events[-1]["turn"]]
    with trace_box:
        for e in events:
            body = {k: v for k, v in e.items() if k not in ("ts", "run_id", "kind", "turn")}
            label = f"{KIND_STYLE.get(e['kind'], '•')} t{e['turn']} **{e['kind']}**"
            if e["kind"] == "tool_call":
                label += f" `{e.get('name')}`" + (f" → ❗{e['error']}" if e.get("error") else "")
            elif e["kind"] == "state_transition":
                label += f" {e['frm']} → {e['to']}"
            elif e["kind"] == "model_call":
                label += f" {e.get('input_tokens')}→{e.get('output_tokens')} tok, {e.get('latency_ms')} ms"
            with st.expander(label):
                st.code(json.dumps(body, indent=1, ensure_ascii=False, default=str)[:4000], language="json")
