"""Side-effect execution: book / cancel / reschedule.

Correctness rules implemented here:
1. Every side effect needs a valid, unconsumed approval whose fingerprint matches the *current* material
   details (re-checked right before the call).
2. Idempotency key = hash(user, action, target, generation). It is written to `pending_ops` (status
   in_flight) BEFORE the tool call — a write-ahead record — so retries, repeated confirmations and
   process restarts all reuse the same key.
3. A timeout (or any ambiguous error) is never treated as failure or success: we consult
   get_booking_status (the source of truth) first, and only retry with the same key if it says nothing
   was booked.
4. Success is reported only after verification.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime

from agent.approval import ApprovalInvalid, Approvals, fingerprint
from mocks.tools import ToolError


@dataclass
class Outcome:
    status: str                  # confirmed | already_done | needs_reapproval | alternatives | unverified | failed
    message: str
    result: dict | None = None
    new_proposal: dict | None = None
    alternatives: list = field(default_factory=list)


def idem_key(*parts) -> str:
    return "idem_" + hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()[:24]


def fmt_slot(slot: str) -> str:
    return datetime.fromisoformat(slot).strftime("%a %d %b, %I:%M %p").replace(" 0", " ")


class BookingExecutor:
    def __init__(self, db, toolbox, approvals: Approvals, tracer, user_id: str, now_fn):
        self.db, self.tools, self.approvals, self.tracer = db, toolbox, approvals, tracer
        self.user_id, self.now = user_id, now_fn

    # ------------------------------------------------------------ pending_ops (write-ahead log)
    def _op_tuple(self, p: dict) -> str:
        if p["action"] == "book":
            return f"{self.user_id}|book|{p['provider_id']}|{p['slot']}"
        if p["action"] == "cancel":
            return f"{self.user_id}|cancel|{p['booking_id']}"
        return f"{self.user_id}|reschedule|{p['booking_id']}|{p['new_slot']}"

    def find_op(self, op_tuple: str, statuses=("in_flight", "confirmed")) -> dict | None:
        q = f"SELECT * FROM pending_ops WHERE op_tuple=? AND status IN ({','.join('?' * len(statuses))}) " \
            "ORDER BY created_at DESC LIMIT 1"
        return self.db.one(q, (op_tuple, *statuses))

    def _new_key(self, op_tuple: str) -> str:
        generation = self.db.one("SELECT COUNT(*) AS n FROM pending_ops WHERE op_tuple=? AND status='released'",
                                 (op_tuple,))["n"]
        return idem_key(op_tuple, generation)

    def _write_ahead(self, key, session_id, action, op_tuple, approval_id, payload) -> None:
        self.db.execute(
            "INSERT INTO pending_ops(idem_key,session_id,user_id,action,op_tuple,approval_id,payload_json,status,"
            "attempts,created_at,updated_at) VALUES(?,?,?,?,?,?,?,'in_flight',0,datetime('now'),datetime('now')) "
            "ON CONFLICT(idem_key) DO UPDATE SET status='in_flight', approval_id=excluded.approval_id, "
            "payload_json=excluded.payload_json, updated_at=datetime('now')",
            (key, session_id, self.user_id, action, op_tuple, approval_id, json.dumps(payload, sort_keys=True)))
        self.tracer.emit("pending_op_written", idem_key=key, action=action, op_tuple=op_tuple)

    def _bump(self, key) -> int:
        self.db.execute("UPDATE pending_ops SET attempts=attempts+1, updated_at=datetime('now') WHERE idem_key=?", (key,))
        return self.db.one("SELECT attempts FROM pending_ops WHERE idem_key=?", (key,))["attempts"]

    def _finish(self, key, status, result=None) -> None:
        self.db.execute("UPDATE pending_ops SET status=?, result_json=?, updated_at=datetime('now') WHERE idem_key=?",
                        (status, json.dumps(result, sort_keys=True) if result is not None else None, key))
        self.tracer.emit("pending_op_closed", idem_key=key, status=status)

    def release_booking(self, booking_id: str) -> None:
        """After a cancellation, the (provider, slot) may legitimately be booked again: bump generation."""
        for op in self.db.all("SELECT * FROM pending_ops WHERE action='book' AND status='confirmed'"):
            if op["result_json"] and json.loads(op["result_json"]).get("booking_id") == booking_id:
                self._finish(op["idem_key"], "released", json.loads(op["result_json"]))

    # ------------------------------------------------------------------ entry point
    def execute(self, session, approval_id: str) -> Outcome:
        p = session.proposal
        if p["action"] == "book":
            return self._book(session, approval_id)
        if p["action"] == "cancel":
            return self._cancel(session, approval_id)
        return self._reschedule(session, approval_id)

    # ------------------------------------------------------------------------- book
    def _book(self, session, approval_id: str) -> Outcome:
        p = session.proposal
        op_tuple = self._op_tuple(p)

        # Duplicate protection across repeated confirmations / restarts.
        existing = self.find_op(op_tuple)
        if existing and existing["status"] == "confirmed":
            self.approvals.consume(approval_id)
            res = json.loads(existing["result_json"])
            self.tracer.emit("duplicate_prevented", idem_key=existing["idem_key"], booking_id=res.get("booking_id"))
            return Outcome("already_done", f"This is already booked (booking {res['booking_id']}). "
                                           "No second booking was made.", res)
        key = existing["idem_key"] if existing else self._new_key(op_tuple)

        # Refresh the hold if it has expired locally; the price may change on re-hold.
        if datetime.fromisoformat(p["hold_expires_at"]) <= self.now():
            refreshed = self._rehold(session)
            if refreshed:
                return refreshed

        try:
            self.approvals.validate(approval_id, p)
        except ApprovalInvalid as e:
            return Outcome("needs_reapproval", f"The details changed since you approved ({e.reason}). "
                                               "Nothing was booked. Please confirm the updated details.",
                           new_proposal=p)

        self._write_ahead(key, session.session_id, "book", op_tuple, approval_id, p)
        return self._attempt_book(session, approval_id, key)

    def _attempt_book(self, session, approval_id: str, key: str) -> Outcome:
        p = session.proposal
        max_attempts = 3
        while True:
            attempts = self._bump(key)
            try:
                res = self.tools.call("book", retry_transient=False, hold_id=p["hold_id"], idempotency_key=key)
            except ToolError as e:
                # Ambiguous: the booking may or may not exist. Ask the source of truth before anything else.
                self.tracer.emit("ambiguous_outcome", tool="book", error=e.code, idem_key=key)
                status = self._status(key)
                if status is None:
                    return Outcome("unverified",
                                   "The booking request timed out and I couldn't verify the result yet. I have NOT "
                                   "confirmed anything, and I will not book twice. I'll re-check before doing anything "
                                   "else.")
                if status.get("status") == "confirmed":
                    self.tracer.emit("reconciled", idem_key=key, verdict="booked_despite_timeout",
                                     booking_id=status["booking_id"])
                    return self._confirmed(session, approval_id, key, status)
                self.tracer.emit("reconciled", idem_key=key, verdict="not_booked", status=status.get("status"))
                if attempts >= max_attempts:
                    self._finish(key, "failed", status)
                    self.approvals.consume(approval_id)
                    return Outcome("failed", "The provider's system kept timing out and I verified that nothing was "
                                             "booked. You can try again or pick another provider.")
                continue  # safe: verified not booked; retry with the SAME key

            if "error" in res:
                err = res["error"]
                if err == "hold_expired" and attempts < max_attempts:
                    # Nothing was booked under this key; re-hold and continue with the same key only if the
                    # material details (incl. price) are unchanged — otherwise fresh approval is required.
                    outcome = self._rehold(session)
                    if outcome:
                        self._finish(key, "failed", res)
                        return outcome
                    continue
                self._finish(key, "failed", res)
                self.approvals.consume(approval_id)
                if err == "slot_taken":
                    return Outcome("alternatives", f"That slot was just taken by someone else, so nothing was booked.",
                                   alternatives=self._alternatives(session))
                return Outcome("failed", f"The booking failed ({err}). Nothing was booked.")

            # Reported success: still verify against the source of truth before claiming it.
            status = self._status(key)
            if status and status.get("status") == "confirmed":
                return self._confirmed(session, approval_id, key, status)
            return Outcome("unverified", "The provider reported success but I couldn't verify it yet, so I'm not "
                                         "calling it confirmed. I'll re-check.")

    def _status(self, key: str) -> dict | None:
        try:
            return self.tools.call("get_booking_status", idempotency_key=key)
        except ToolError:
            return None

    def _confirmed(self, session, approval_id, key, status) -> Outcome:
        p = session.proposal
        result = {"booking_id": status["booking_id"], "provider_id": p["provider_id"],
                  "provider_name": p.get("provider_name"), "slot": status["slot"], "price": status["price"],
                  "service": p.get("service"), "verified": True}
        self._finish(key, "confirmed", result)
        self.approvals.consume(approval_id)
        return Outcome("confirmed", f"Booked and verified: {p.get('provider_name')} ({p.get('service')}) on "
                                    f"{fmt_slot(result['slot'])} for ₹{result['price']:,}. Booking ID {result['booking_id']}.",
                       result)

    def _rehold(self, session) -> Outcome | None:
        """Re-hold the same provider+slot. Returns None if the material details are unchanged (approval still
        valid, caller continues), or an Outcome asking for fresh approval."""
        p = session.proposal
        old_fp = fingerprint(p)
        hold = self.tools.call("hold_slot", provider_id=p["provider_id"], slot=p["slot"])
        if "error" in hold:
            self.approvals.invalidate_session(session.session_id, "hold_lost")
            return Outcome("alternatives", "Your held slot expired and is no longer available. Nothing was booked.",
                           alternatives=self._alternatives(session))
        p.update(hold_id=hold["hold_id"], hold_expires_at=hold["expires_at"])
        if hold["quoted_price"] != p["price"]:
            old_price = p["price"]
            p["price"] = hold["quoted_price"]
            p["warnings"] = p.get("warnings", []) + [f"price changed from ₹{old_price:,} to ₹{p['price']:,} on re-hold"]
            self.approvals.invalidate_session(session.session_id, "price_changed")
            assert fingerprint(p) != old_fp
            return Outcome("needs_reapproval", f"The hold expired and the provider re-quoted ₹{p['price']:,} "
                                               f"(was ₹{old_price:,}). Nothing was booked.", new_proposal=p)
        self.tracer.emit("rehold", hold_id=hold["hold_id"], price_unchanged=True)
        return None

    def _alternatives(self, session) -> list[dict]:
        p = session.proposal or {}
        alts = []
        for c in session.candidates:
            for s in c.get("matching_slots", []):
                if (c["provider_id"], s) != (p.get("provider_id"), p.get("slot")):
                    alts.append({"provider_id": c["provider_id"], "name": c["name"], "slot": s, "price": c["price"]})
                    break
        return alts[:3]

    # ----------------------------------------------------------------------- cancel
    def _cancel(self, session, approval_id: str) -> Outcome:
        p = session.proposal
        try:
            self.approvals.validate(approval_id, p)
        except ApprovalInvalid as e:
            return Outcome("needs_reapproval", f"Details changed since approval ({e.reason}). Nothing was cancelled.",
                           new_proposal=p)
        op_tuple = self._op_tuple(p)
        existing = self.find_op(op_tuple)
        if existing and existing["status"] == "confirmed":
            self.approvals.consume(approval_id)
            return Outcome("already_done", f"Booking {p['booking_id']} is already cancelled.",
                           json.loads(existing["result_json"]))
        key = existing["idem_key"] if existing else self._new_key(op_tuple)
        self._write_ahead(key, session.session_id, "cancel", op_tuple, approval_id, p)
        self._bump(key)
        try:
            # 503s are retried inside the toolbox with the same idempotency key.
            res = self.tools.call("cancel", booking_id=p["booking_id"], idempotency_key=key)
        except ToolError as e:
            self.tracer.emit("ambiguous_outcome", tool="cancel", error=e.code, idem_key=key)
            res = {"error": e.code}
        verified = self._booking_status(p["booking_id"]) == "cancelled"
        if verified:
            result = {"booking_id": p["booking_id"], "status": "cancelled", "verified": True}
            self._finish(key, "confirmed", result)
            self.approvals.consume(approval_id)
            self.release_booking(p["booking_id"])
            return Outcome("confirmed", f"Cancelled and verified: booking {p['booking_id']} "
                                        f"({p.get('provider_name')}, {fmt_slot(p['slot'])}).", result)
        self._finish(key, "failed", res)
        self.approvals.consume(approval_id)
        return Outcome("failed", f"I couldn't cancel booking {p['booking_id']} after retries "
                                 f"({res.get('error', 'unknown')}). It is still active.")

    def _booking_status(self, booking_id: str) -> str | None:
        try:
            for b in self.tools.call("list_bookings", user_id=self.user_id)["bookings"]:
                if b["booking_id"] == booking_id:
                    return b["status"] if b["status"] != "confirmed" else f"confirmed@{b['slot']}"
        except ToolError:
            return None
        return None

    # ------------------------------------------------------------------- reschedule
    def _reschedule(self, session, approval_id: str) -> Outcome:
        p = session.proposal
        try:
            self.approvals.validate(approval_id, p)
        except ApprovalInvalid as e:
            return Outcome("needs_reapproval", f"Details changed since approval ({e.reason}). Nothing was changed.",
                           new_proposal=p)
        op_tuple = self._op_tuple(p)
        existing = self.find_op(op_tuple)
        if existing and existing["status"] == "confirmed":
            self.approvals.consume(approval_id)
            return Outcome("already_done", "That reschedule is already done.", json.loads(existing["result_json"]))
        key = existing["idem_key"] if existing else self._new_key(op_tuple)
        self._write_ahead(key, session.session_id, "reschedule", op_tuple, approval_id, p)
        self._bump(key)
        try:
            res = self.tools.call("reschedule", booking_id=p["booking_id"], slot=p["new_slot"], idempotency_key=key)
        except ToolError as e:
            self.tracer.emit("ambiguous_outcome", tool="reschedule", error=e.code, idem_key=key)
            res = {"error": e.code}
        if res.get("error") == "slot_conflict":
            self._finish(key, "failed", res)
            self.approvals.consume(approval_id)
            return Outcome("alternatives", f"{fmt_slot(p['new_slot'])} conflicts with the provider's schedule, so "
                                           f"your booking is unchanged ({fmt_slot(p['slot'])}).",
                           alternatives=p.get("alternative_slots", []))
        if self._booking_status(p["booking_id"]) == f"confirmed@{p['new_slot']}":
            result = {"booking_id": p["booking_id"], "slot": p["new_slot"], "verified": True}
            self._finish(key, "confirmed", result)
            self.approvals.consume(approval_id)
            return Outcome("confirmed", f"Rescheduled and verified: booking {p['booking_id']} moved from "
                                        f"{fmt_slot(p['slot'])} to {fmt_slot(p['new_slot'])}.", result)
        self._finish(key, "failed", res)
        self.approvals.consume(approval_id)
        return Outcome("failed", f"The reschedule didn't go through ({res.get('error', 'unverified')}). "
                                 "Your original booking is unchanged.")

    # --------------------------------------------------------------------- recovery
    def recover(self) -> list[dict]:
        """On startup: reconcile every in-flight side effect against the source of truth."""
        recovered = []
        for op in self.db.all("SELECT * FROM pending_ops WHERE status='in_flight' AND user_id=?", (self.user_id,)):
            payload = json.loads(op["payload_json"])
            if op["action"] == "book":
                status = self._status(op["idem_key"])
                if status and status.get("status") == "confirmed":
                    result = {"booking_id": status["booking_id"], "provider_id": payload["provider_id"],
                              "provider_name": payload.get("provider_name"), "slot": status["slot"],
                              "price": status["price"], "service": payload.get("service"), "verified": True}
                    self._finish(op["idem_key"], "confirmed", result)
                    self.approvals.consume(op["approval_id"])
                    recovered.append({"op": op, "verdict": "confirmed", "result": result, "proposal": payload})
                elif status is not None:
                    self._finish(op["idem_key"], "failed", status)
                    self.approvals.invalidate_one(op["approval_id"], "interrupted")
                    recovered.append({"op": op, "verdict": "not_booked", "result": status, "proposal": payload})
                else:
                    recovered.append({"op": op, "verdict": "unknown", "proposal": payload})
            else:
                bstatus = self._booking_status(payload["booking_id"])
                done = (op["action"] == "cancel" and bstatus == "cancelled") or \
                       (op["action"] == "reschedule" and bstatus == f"confirmed@{payload['new_slot']}")
                self._finish(op["idem_key"], "confirmed" if done else "failed", {"booking_status": bstatus})
                recovered.append({"op": op, "verdict": "confirmed" if done else "not_applied", "proposal": payload})
            self.tracer.emit("recovered_op", idem_key=op["idem_key"], action=op["action"],
                             verdict=recovered[-1]["verdict"])
        return recovered
