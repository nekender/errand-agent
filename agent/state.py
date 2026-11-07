"""Explicit workflow state machine. The session is persisted after every transition."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from enum import Enum

from agent.slots import Slots


class Phase(str, Enum):
    GATHERING = "GATHERING"                  # collecting required slots
    COMPARING = "COMPARING"                  # have search results, choosing
    AWAITING_APPROVAL = "AWAITING_APPROVAL"  # a concrete proposal (held slot / cancel / reschedule) is shown
    EXECUTING = "EXECUTING"                  # side effect in flight (book/cancel/reschedule)
    VERIFYING = "VERIFYING"                  # outcome ambiguous, consulting source of truth
    DONE = "DONE"
    FAILED = "FAILED"


P = Phase
TRANSITIONS: dict[Phase, set[Phase]] = {
    P.GATHERING: {P.GATHERING, P.COMPARING, P.AWAITING_APPROVAL, P.FAILED},
    P.COMPARING: {P.GATHERING, P.COMPARING, P.AWAITING_APPROVAL, P.FAILED},
    P.AWAITING_APPROVAL: {P.GATHERING, P.COMPARING, P.AWAITING_APPROVAL, P.EXECUTING, P.FAILED},
    P.EXECUTING: {P.VERIFYING, P.DONE, P.FAILED, P.COMPARING, P.AWAITING_APPROVAL},
    P.VERIFYING: {P.EXECUTING, P.DONE, P.FAILED, P.COMPARING, P.AWAITING_APPROVAL},
    P.DONE: {P.GATHERING, P.COMPARING, P.AWAITING_APPROVAL, P.DONE},
    P.FAILED: {P.GATHERING, P.COMPARING, P.AWAITING_APPROVAL, P.FAILED},
}


class IllegalTransition(Exception):
    pass


@dataclass
class Session:
    session_id: str
    user_id: str
    phase: Phase = Phase.GATHERING
    slots: Slots = field(default_factory=Slots)
    candidates: list[dict] = field(default_factory=list)   # compact last search results
    proposal: dict | None = None                            # the exact action shown for approval
    last_result: dict | None = None                         # last verified side-effect result
    turn: int = 0

    def to_json(self) -> str:
        d = asdict(self)
        d["phase"] = self.phase.value
        return json.dumps(d, sort_keys=True, default=str)

    @classmethod
    def from_json(cls, s: str) -> "Session":
        d = json.loads(s)
        d["phase"] = Phase(d["phase"])
        d["slots"] = Slots.from_dict(d["slots"])
        return cls(**d)


class StateStore:
    def __init__(self, db, tracer):
        self.db = db
        self.tracer = tracer

    def load(self, session_id: str, user_id: str) -> Session:
        row = self.db.one("SELECT state_json FROM sessions WHERE session_id=?", (session_id,))
        return Session.from_json(row["state_json"]) if row else Session(session_id, user_id)

    def save(self, s: Session) -> None:
        self.db.execute(
            "INSERT INTO sessions(session_id,user_id,state_json,updated_at) VALUES(?,?,?,datetime('now')) "
            "ON CONFLICT(session_id) DO UPDATE SET state_json=excluded.state_json, updated_at=excluded.updated_at",
            (s.session_id, s.user_id, s.to_json()))

    def transition(self, s: Session, to: Phase, reason: str) -> None:
        if to not in TRANSITIONS[s.phase]:
            self.tracer.emit("illegal_transition", frm=s.phase.value, to=to.value, reason=reason)
            raise IllegalTransition(f"{s.phase.value} -> {to.value}")
        self.tracer.emit("state_transition", frm=s.phase.value, to=to.value, reason=reason)
        s.phase = to
        self.save(s)
