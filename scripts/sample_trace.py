"""Produce traces/sample_timeout_after_success.jsonl: one full booking where book() times out after the
provider committed. Uses Claude when credentials are available (--llm claude), else the scripted model."""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv  # noqa: E402

from agent.db import DB  # noqa: E402
from agent.memory import Memory  # noqa: E402
from agent.orchestrator import Agent  # noqa: E402
from agent.seed import seed_conversation, seed_memories  # noqa: E402
from mocks.tools import MockWorld, Scenario  # noqa: E402

load_dotenv(ROOT / ".env")
ap = argparse.ArgumentParser()
ap.add_argument("--llm", choices=["claude", "scripted"], default="scripted")
args = ap.parse_args()

world = MockWorld(Scenario(book_outcomes=["timeout_booked"]))
db = DB()
seed_memories(Memory(db, "u1", now_fn=world.clock.now))
seed_conversation(db, "u1")
if args.llm == "claude":
    from agent.llm import AnthropicLLM, RecordingLLM
    llm = RecordingLLM(AnthropicLLM(), ROOT / "evals/fixtures/claude_timeout_after_success.json")
else:
    from agent.llm import ScriptedLLM
    from evals.harness import booking_steps
    llm = ScriptedLLM(booking_steps())

out = ROOT / "traces"
agent = Agent(world, llm, db=db, session_id=f"sample-{args.llm}", trace_dir=out, run_id=f"sample-{args.llm}")
if agent.tracer.path.exists():
    agent.tracer.path.unlink()
for msg in ["Find a plumber for my kitchen tap this week, mornings only, under ₹1,500.", "yes"]:
    print(f"you> {msg}\nagent> {agent.handle(msg)}\n")
    if agent.session.phase.value == "GATHERING" and args.llm == "claude":
        print("(model asked a question; answer it in cli.py chat instead)")
        break
dest = out / "sample_timeout_after_success.jsonl"
shutil.copy(agent.tracer.path, dest)
print("trace:", dest, "| totals:", agent.tracer.totals())
