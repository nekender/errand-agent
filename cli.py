"""Terminal chat with the Errand agent.

  python cli.py chat [--scripted-demo] [--book-outcome timeout_booked] [--db errand.db] [--fresh]
  python cli.py cancel-event <booking_id>     # simulate a provider cancellation (picked up on next prompt)

Commands inside chat:  /trace  /memory  /state  /tick  /advance <minutes>  /quit
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

from agent import proactive
from agent.db import DB
from agent.limits import Limits
from agent.memory import Memory
from agent.orchestrator import Agent
from agent.seed import seed_conversation, seed_memories
from mocks.tools import MockWorld, Scenario

ROOT = Path(__file__).parent
STATE = ROOT / ".state"


def build(args) -> Agent:
    STATE.mkdir(exist_ok=True)
    if args.fresh:
        for f in STATE.glob("*"):
            f.unlink()
    db = DB(STATE / "agent.db")
    world = MockWorld(Scenario(seed=args.seed, book_outcomes=[args.book_outcome] if args.book_outcome else []),
                      persist_path=STATE / "world.json")
    if db.one("SELECT COUNT(*) AS n FROM facts")["n"] == 0:
        seed_memories(Memory(db, "u1", now_fn=world.clock.now))
        seed_conversation(db, "u1")
    if args.llm == "claude":
        from agent.llm import AnthropicLLM, RecordingLLM
        llm = AnthropicLLM()
        if args.record:
            llm = RecordingLLM(llm, args.record)
    else:
        from agent.llm import ReplayLLM
        llm = ReplayLLM(args.replay)
    return Agent(world, llm, db=db, session_id=args.session, limits=Limits(retry_backoff_s=0.2),
                 trace_dir=ROOT / "traces", context_budget=args.context_budget)


def chat(args) -> None:
    agent = build(args)
    print(f"Errand agent — model={agent.llm.name} session={agent.session_id} trace={agent.tracer.path}")
    print("Type /quit to exit, /trace for the last turn's trace, /memory, /state, /tick, /advance N.\n")
    for m in agent.drain_outbox():
        print(f"agent (notice)> {m}\n")
    while True:
        for m in proactive.tick(agent):
            print(f"agent (proactive)> {m}\n")
        try:
            text = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not text:
            continue
        if text == "/quit":
            break
        if text == "/trace":
            for e in agent.tracer.events:
                if e["turn"] == agent.session.turn:
                    print(json.dumps({k: v for k, v in e.items() if k not in ("ts", "run_id")}, ensure_ascii=False)[:300])
            continue
        if text == "/memory":
            print(json.dumps(agent.memory.inspect(), indent=1, ensure_ascii=False))
            continue
        if text == "/state":
            print(agent.session.to_json())
            continue
        if text == "/tick":
            continue
        if text.startswith("/advance"):
            agent.world.clock.advance(minutes=int(text.split()[1]))
            print(f"clock: {agent.world.clock.now()}")
            continue
        print(f"\nagent> {agent.handle(text)}\n")
    print("totals:", agent.tracer.totals())


def main() -> None:
    load_dotenv(ROOT / ".env")
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("chat")
    c.add_argument("--llm", choices=["claude", "replay"], default="claude")
    c.add_argument("--replay", help="fixture JSON for --llm replay")
    c.add_argument("--record", help="save Claude responses to this fixture path")
    c.add_argument("--book-outcome", choices=["success", "slot_taken", "timeout_booked", "timeout_not_booked"])
    c.add_argument("--seed", type=int, default=42)
    c.add_argument("--session", default="cli")
    c.add_argument("--context-budget", type=int, default=3000)
    c.add_argument("--fresh", action="store_true", help="wipe .state/ (agent DB + mock world) first")
    e = sub.add_parser("cancel-event")
    e.add_argument("booking_id")
    args = ap.parse_args()
    if args.cmd == "chat":
        if args.llm == "claude" and not (os.getenv("ANTHROPIC_API_KEY") or os.getenv("ANTHROPIC_AUTH_TOKEN")):
            print("note: no ANTHROPIC_API_KEY in env/.env; the SDK will try an `ant auth login` profile.", file=sys.stderr)
        chat(args)
    else:
        world = MockWorld(persist_path=STATE / "world.json")
        world.emit_provider_cancel(args.booking_id)
        print(f"provider cancelled {args.booking_id}; the running agent will notice on its next check")


if __name__ == "__main__":
    main()
