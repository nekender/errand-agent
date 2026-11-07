"""Approval enforcement in application code.

* Approvals are created only by the orchestrator, only in response to an explicit affirmative *user
  message*, only while a proposal is on screen. The model has no tool that can create one.
* An approval is bound to a fingerprint of the material details (action, provider, slot, price,
  booking). Executors recompute the fingerprint from the *current* hold/booking; any difference
  invalidates the approval.
* Approvals are single-use.
"""
from __future__ import annotations

import hashlib
import json
import re
import uuid

MATERIAL_FIELDS = ("action", "provider_id", "slot", "price", "booking_id", "new_slot")

_AFFIRM = re.compile(
    r"^\s*(yes|y|yeah|yep|yup|confirm(ed)?|go ahead|book it|do it|please do|ok(ay)?|sure|approve(d)?)"
    r"(,?\s*(please|go ahead|book it|do it|confirm(ed)?|thanks?|thank you))*[\s.!]*$", re.I)
_DECLINE = re.compile(r"^\s*(no|n|nope|nah|don'?t|do not|stop|not now|never ?mind|hold off)\b", re.I)


def classify_reply(text: str) -> str:
    """Deterministic approval classifier. Anything that is not an unambiguous yes/no is 'other' and goes
    back through the model, which cannot approve. We deliberately prefer re-asking over guessing."""
    if _AFFIRM.match(text or ""):
        return "yes"
    if _DECLINE.match(text or ""):
        return "no"
    return "other"


def material(proposal: dict) -> dict:
    return {k: proposal.get(k) for k in MATERIAL_FIELDS}


def fingerprint(proposal: dict) -> str:
    return hashlib.sha256(json.dumps(material(proposal), sort_keys=True).encode()).hexdigest()[:20]


class ApprovalInvalid(Exception):
    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


class Approvals:
    def __init__(self, db, tracer):
        self.db, self.tracer = db, tracer

    def grant(self, session_id: str, proposal: dict, user_text: str) -> str:
        approval_id = f"apr_{uuid.uuid4().hex[:10]}"
        fp = fingerprint(proposal)
        self.db.execute(
            "INSERT INTO approvals(approval_id,session_id,action,fingerprint,payload_json,status,created_at) "
            "VALUES(?,?,?,?,?,'granted',datetime('now'))",
            (approval_id, session_id, proposal["action"], fp, json.dumps(material(proposal), sort_keys=True)))
        self.tracer.emit("approval_granted", approval_id=approval_id, fingerprint=fp,
                         material=material(proposal), user_text=user_text)
        return approval_id

    def validate(self, approval_id: str, current: dict) -> dict:
        row = self.db.one("SELECT * FROM approvals WHERE approval_id=?", (approval_id,))
        if not row:
            raise ApprovalInvalid("no_such_approval")
        if row["status"] != "granted":
            raise ApprovalInvalid(f"approval_{row['status']}")
        fp = fingerprint(current)
        if fp != row["fingerprint"]:
            self.invalidate_one(approval_id, "material_change")
            self.tracer.emit("approval_invalidated", approval_id=approval_id, reason="material_change",
                             approved=json.loads(row["payload_json"]), current=material(current))
            raise ApprovalInvalid("material_change")
        return row

    def consume(self, approval_id: str) -> None:
        self.db.execute("UPDATE approvals SET status='consumed', consumed_at=datetime('now') "
                        "WHERE approval_id=? AND status='granted'", (approval_id,))
        self.tracer.emit("approval_consumed", approval_id=approval_id)

    def invalidate_one(self, approval_id: str, reason: str) -> None:
        self.db.execute("UPDATE approvals SET status=? WHERE approval_id=? AND status='granted'",
                        (f"invalidated:{reason}", approval_id))

    def invalidate_session(self, session_id: str, reason: str) -> int:
        n = self.db.execute("UPDATE approvals SET status=? WHERE session_id=? AND status='granted'",
                            (f"invalidated:{reason}", session_id)).rowcount
        if n:
            self.tracer.emit("approval_invalidated", session_id=session_id, reason=reason, count=n)
        return n
