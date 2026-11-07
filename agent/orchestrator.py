"""The Errand agent: deterministic orchestration around a model that can only read and propose."""
from __future__ import annotations

import json
import re
import uuid
from datetime import date, datetime
from pathlib import Path

from agent.approval import Approvals, classify_reply
from agent.booking import BookingExecutor, Outcome, fmt_slot
from agent.context import ContextBuilder
from agent.db import DB
from agent.limits import Budget, BudgetExceeded, Limits
from agent.llm import LLM
from agent.memory import Memory
from agent.sanitize import sanitize_provider
from agent.state import Phase, StateStore
from agent.toolbox import Toolbox
from agent.tools_api import MODEL_TOOLS, TOOL_NAMES
from mocks.tools import MockWorld, ToolError

# First-person claims that a side effect happened. The model is never allowed to make these; only
# code-generated messages (after verification) may.
CLAIM_RX = re.compile(
    r"\b(i|we)\s*('ve|'ll| have| just| successfully)*\s*(booked|cancell?ed|rescheduled|confirmed the booking)\b"
    r"|(?<!nothing )(?<!not )\b(has|have) been (booked|cancell?ed|rescheduled)\b|\byou'?re (all set|booked)\b", re.I)


class EndTurn(Exception):
    """Raised by a tool handler when the turn must end with a code-generated message (e.g. a proposal)."""

    def __init__(self, message: str, result: dict):
        super().__init__(message)
        self.message, self.result = message, result


class Agent:
    def __init__(self, world: MockWorld, llm: LLM, db: DB | None = None, user_id: str = "u1",
                 session_id: str | None = None, limits: Limits | None = None, trace_dir: str | Path | None = "traces",
                 context_budget: int = 3000, run_id: str | None = None):
        from agent.trace import Tracer

        self.world, self.llm, self.user_id = world, llm, user_id
        self.db = db or DB()
        self.session_id = session_id or f"s_{uuid.uuid4().hex[:8]}"
        self.limits = limits or Limits()
        self.tracer = Tracer(run_id or f"{self.session_id}-{uuid.uuid4().hex[:6]}", trace_dir)
        self.now = world.clock.now
        self.store = StateStore(self.db, self.tracer)
        self.session = self.store.load(self.session_id, user_id)
        self.memory = Memory(self.db, user_id, now_fn=self.now)
        self.toolbox = Toolbox(world, self.tracer, self.limits)
        self.approvals = Approvals(self.db, self.tracer)
        self.booking = BookingExecutor(self.db, self.toolbox, self.approvals, self.tracer, user_id, self.now)
        self.context = ContextBuilder(self.db, self.memory, self.tracer, budget_tokens=context_budget, now_fn=self.now)
        self.tracer.emit("run_start", session_id=self.session_id, user_id=user_id, model=llm.name,
                         phase=self.session.phase.value, limits=self.limits.__dict__)
        self._verified_this_turn = False
        self._turn_text = ""
        self._recover()

    # ================================================================== public API
    def handle(self, text: str) -> str:
        s = self.session
        s.turn += 1
        self.tracer.turn = s.turn
        self._verified_this_turn = False
        self._turn_text = text
        self._save_message("user", text)
        self.tracer.emit("turn_start", user_text=text, phase=s.phase.value)
        budget = Budget(self.limits)
        self.toolbox.budget = budget
        outcome = "ok"
        try:
            reply = self._handle(text, budget)
        except BudgetExceeded as e:
            outcome = f"limit:{e.which}"
            reply = (f"I had to stop: this request hit my {e.which.replace('_', ' ')} limit ({e.limit}). "
                     + self._safe_status() + " Could you rephrase or narrow the request?")
        except Exception as e:  # never fail silently
            outcome = f"error:{type(e).__name__}"
            self.tracer.emit("error", error=type(e).__name__, message=str(e))
            reply = "Something went wrong on my side. " + self._safe_status()
        finally:
            self.toolbox.budget = None
        reply = self._guard_claims(reply)
        self._save_message("assistant", reply)
        self.store.save(s)
        self.tracer.emit("turn_end", outcome=outcome, phase=s.phase.value, reply=reply, **self.tracer.totals())
        return reply

    def drain_outbox(self) -> list[str]:
        rows = self.db.all("SELECT id, content FROM outbox WHERE session_id=? AND delivered=0 ORDER BY id",
                           (self.session_id,))
        for r in rows:
            self.db.execute("UPDATE outbox SET delivered=1 WHERE id=?", (r["id"],))
        return [r["content"] for r in rows]

    def notify(self, text: str, channel: str) -> None:
        """Agent-initiated message (proactive follow-up, recovery notice)."""
        self._save_message("assistant", text)
        self.db.execute("INSERT INTO outbox(session_id,content,created_at) VALUES(?,?,datetime('now'))",
                        (self.session_id, text))
        self.tracer.emit("agent_notification", channel=channel, text=text)

    # ================================================================ turn handling
    def _handle(self, text: str, budget: Budget) -> str:
        s = self.session
        if s.phase == Phase.VERIFYING:
            note = self._reconcile_in_flight()
            if note:
                return note

        reply_kind = classify_reply(text)
        if s.phase == Phase.AWAITING_APPROVAL and s.proposal:
            if reply_kind == "yes":
                return self._approve_and_execute(text)
            if reply_kind == "no":
                self.approvals.invalidate_session(s.session_id, "user_declined")
                s.proposal = None
                self.store.transition(s, Phase.COMPARING if s.candidates else Phase.GATHERING, "user declined")
                return "Okay, I won't go ahead with that. Nothing was booked or changed. Want a different option?"
        elif reply_kind == "yes" and s.phase == Phase.DONE and s.last_result:
            self.tracer.emit("duplicate_prevented", reason="confirmation_after_done", last=s.last_result)
            return (f"That's already done ({self._describe(s.last_result)}). "
                    "I haven't made any new booking or change.")
        elif reply_kind == "yes" and not s.proposal:
            self.tracer.emit("approval_ignored", reason="no_pending_proposal")
        return self._model_loop(text, budget)

    def _model_loop(self, text: str, budget: Budget) -> str:
        system, messages, _ = self.context.build(self.session, text)
        while True:
            budget.model_call()
            resp = self.llm.complete(system, messages, MODEL_TOOLS)
            self.tracer.emit("model_call", model=resp.model, input_tokens=resp.input_tokens,
                             output_tokens=resp.output_tokens, cache_read_tokens=resp.cache_read_tokens,
                             cost_usd=round(resp.cost_usd, 6), latency_ms=resp.latency_ms,
                             stop_reason=resp.stop_reason, text=resp.text,
                             tool_calls=[{"name": c.name, "args": c.args} for c in resp.tool_calls])
            messages.append({"role": "assistant", "content": resp.assistant_content})
            if not resp.tool_calls:
                return resp.text or "Sorry, I didn't catch that. Could you rephrase?"
            results, end = [], None
            for c in resp.tool_calls:
                if end is not None:
                    results.append(_tool_result(c.id, {"skipped": "turn ended by a proposal"}, False))
                    continue
                try:
                    out = self._dispatch(c.name, c.args, budget)
                except EndTurn as e:
                    end, out = e, e.result
                except BudgetExceeded:
                    raise
                except Exception as e:
                    out = {"error": f"{type(e).__name__}: {e}"}
                    self.tracer.emit("tool_handler_error", name=c.name, error=str(e))
                    budget.tool_error()
                results.append(_tool_result(c.id, out, "error" in out))
            messages.append({"role": "user", "content": results})
            if end is not None:
                return end.message

    # ================================================================== dispatch
    def _dispatch(self, name: str, args: dict, budget: Budget) -> dict:
        if name not in TOOL_NAMES:
            return {"error": f"unknown tool {name}"}
        self.tracer.emit("model_tool", name=name, args=args)
        return getattr(self, f"_t_{name}")(**args)

    # -- request ------------------------------------------------------------------
    def _t_update_request(self, **fields) -> dict:
        s = self.session
        if fields.get("budget_max") is not None and not self._grounded_number(fields["budget_max"]):
            # Defence in depth against injected intent changes: a budget must come from the user's own words
            # (this turn) or their stored preferences, never from tool output.
            self.tracer.emit("ungrounded_update_rejected", field="budget_max", value=fields["budget_max"])
            return {"error": f"budget_max={fields['budget_max']} was not stated by the user; ask the user instead"}
        try:
            changed = s.slots.merge(fields)
        except ValueError as e:
            return {"error": str(e)}
        filled = self._apply_memory_defaults()
        if changed and s.proposal and s.proposal["action"] == "book" and \
                set(changed) & {"service", "area", "date_from", "date_to", "earliest_hour", "latest_hour",
                                "budget_max", "provider_id"}:
            self.approvals.invalidate_session(s.session_id, "request_changed")
            s.proposal = None
            self.store.transition(s, Phase.GATHERING, f"request changed: {changed}")
        if {"service", "area", "date_from", "date_to"} & set(changed):
            s.candidates = []
            if s.phase == Phase.COMPARING:
                self.store.transition(s, Phase.GATHERING, f"search inputs changed: {changed}")
        self.tracer.emit("slots_updated", changed=changed, filled_from_memory=filled, slots=s.slots.to_dict())
        self.store.save(s)
        return {"request": s.slots.to_dict(), "changed": changed, "filled_from_memory": filled,
                "missing": s.slots.missing()}

    def _grounded_number(self, value) -> bool:
        try:
            v = int(float(str(value).replace(",", "")))
        except ValueError:
            return False
        text = (self._turn_text or "").replace(",", "").lower()
        nums = {int(float(n)) for n in re.findall(r"\d+(?:\.\d+)?", text)}
        nums |= {int(float(n) * 1000) for n in re.findall(r"(\d+(?:\.\d+)?)\s*k\b", text)}
        remembered = {str(f["value"]) for f in self.memory.eligible() if f["key"] == "budget_max"}
        return v in nums or str(v) in remembered

    def _apply_memory_defaults(self) -> list[str]:
        sl, filled = self.session.slots, []
        defaults = {"area": self.memory.get("preferred_area"), "earliest_hour": self.memory.get("earliest_hour"),
                    "latest_hour": self.memory.get("latest_hour")}
        if sl.service:
            defaults["budget_max"] = self.memory.get("budget_max", sl.service)
        for k, v in defaults.items():
            if v is not None and getattr(sl, k) is None:
                sl.merge({k: v})
                sl.from_memory.append(k)
                filled.append(k)
        return filled

    def _preferred_provider(self) -> str | None:
        sl = self.session.slots
        if sl.provider_id:
            return sl.provider_id
        v = self.memory.get("preferred_provider", sl.service) if sl.service else None
        return v.split()[0] if v else None

    def _t_search_providers(self) -> dict:
        s, sl = self.session, self.session.slots
        missing = [m for m in sl.missing() if m != "budget_max"]
        if missing:
            return {"error": f"cannot search yet; ask the user for: {missing}"}
        res = self.toolbox.call("search_providers", service=sl.service, area=sl.area,
                                date_range={"start": sl.date_from, "end": sl.date_to})
        pref = self._preferred_provider()
        cands = []
        for p in res.get("providers", []):
            matching = [x for x in p["available_slots"] if self._slot_ok(x)]
            cands.append({"provider_id": p["provider_id"], "name": p["name"], "area": p["area"], "price": p["price"],
                          "rating": p["rating"], "attributes": p["attributes"],
                          "within_budget": sl.budget_max is None or p["price"] <= sl.budget_max,
                          "preferred": p["provider_id"] == pref, "matching_slots": matching[:4],
                          "all_slots": p["available_slots"]})
        cands.sort(key=lambda c: (not c["preferred"] if sl.provider_id else False, not c["within_budget"],
                                  not c["matching_slots"], not c["preferred"], -c["rating"], c["price"]))
        s.candidates = cands
        if s.phase != Phase.COMPARING:
            self.store.transition(s, Phase.COMPARING, f"{len(cands)} providers found")
        return {"providers": [{k: v for k, v in c.items() if k != "all_slots"} for c in cands],
                "note": "matching_slots respect the user's hours and dates"}

    def _slot_ok(self, slot: str) -> bool:
        sl, dt = self.session.slots, datetime.fromisoformat(slot)
        if sl.date_from and dt.date() < date.fromisoformat(sl.date_from):
            return False
        if sl.date_to and dt.date() > date.fromisoformat(sl.date_to):
            return False
        if sl.earliest_hour is not None and dt.hour < sl.earliest_hour:
            return False
        if sl.latest_hour is not None and dt.hour > sl.latest_hour:
            return False
        return True

    def _t_get_provider(self, provider_id: str) -> dict:
        p = self.toolbox.call("get_provider", provider_id=provider_id)
        if "error" in p:
            return p
        safe, findings = sanitize_provider(p)
        if findings:
            self.tracer.emit("injection_detected", provider_id=provider_id, findings=findings)
        return safe

    def _t_list_bookings(self) -> dict:
        return self.toolbox.call("list_bookings", user_id=self.user_id)

    # -- proposals (end the turn with a code-rendered confirmation request) -----------
    def _t_propose_booking(self, provider_id: str, slot: str) -> dict:
        s, sl = self.session, self.session.slots
        cand = next((c for c in s.candidates if c["provider_id"] == provider_id), None)
        if not cand:
            return {"error": "provider not in current search results; search first"}
        if not self._slot_ok(slot):
            return {"error": "slot violates the user's date/time constraints; pick a matching slot or ask the user "
                             "to change their constraints", "matching_slots": cand["matching_slots"]}
        if slot not in cand["all_slots"]:
            return {"error": f"slot {slot} is not available for {provider_id}", "matching_slots": cand["matching_slots"]}
        if sl.missing():
            return {"error": f"ask the user for: {sl.missing()} before proposing"}
        hold = self.toolbox.call("hold_slot", provider_id=provider_id, slot=slot)
        if "error" in hold:
            return {"error": f"could not hold slot: {hold['error']}"}
        warnings = []
        if hold["quoted_price"] != cand["price"]:
            warnings.append(f"quoted price ₹{hold['quoted_price']:,} differs from the listed ₹{cand['price']:,}")
        if sl.budget_max is not None and hold["quoted_price"] > sl.budget_max:
            self.tracer.emit("proposal_rejected", reason="over_budget", quoted=hold["quoted_price"], budget=sl.budget_max)
            return {"error": f"held price ₹{hold['quoted_price']} exceeds the user's budget ₹{sl.budget_max}. Tell the "
                             "user and ask whether to raise the budget or choose another provider."}
        proposal = {"action": "book", "provider_id": provider_id, "provider_name": cand["name"],
                    "service": sl.service, "slot": slot, "price": hold["quoted_price"], "listed_price": cand["price"],
                    "hold_id": hold["hold_id"], "hold_expires_at": hold["expires_at"], "booking_id": None,
                    "new_slot": None, "warnings": warnings}
        return self._present(proposal)

    def _t_propose_cancel(self, booking_id: str) -> dict:
        b = self._find_booking(booking_id)
        if not b:
            return {"error": "no such booking for this user"}
        if b["status"] != "confirmed" or datetime.fromisoformat(b["slot"]) <= self.now():
            return {"error": f"booking is not an upcoming confirmed booking (status {b['status']})"}
        return self._present({"action": "cancel", "booking_id": booking_id, "provider_id": b["provider_id"],
                              "provider_name": b["provider_name"], "service": b["service"], "slot": b["slot"],
                              "price": b["price"], "new_slot": None, "warnings": []})

    def _t_propose_reschedule(self, booking_id: str, new_slot: str) -> dict:
        b = self._find_booking(booking_id)
        if not b:
            return {"error": "no such booking for this user"}
        if b["status"] != "confirmed" or datetime.fromisoformat(b["slot"]) <= self.now():
            return {"error": f"booking is not an upcoming confirmed booking (status {b['status']})"}
        try:
            dt = datetime.fromisoformat(new_slot)
        except ValueError:
            return {"error": "new_slot must be YYYY-MM-DDTHH:MM"}
        if dt <= self.now():
            return {"error": "new_slot is in the past"}
        new_slot = dt.isoformat(timespec="minutes")
        if new_slot == b["slot"]:
            return {"error": "new_slot equals the current slot"}
        hours = self.world.provider_hours(b["provider_id"])  # provider's published working hours
        alts = [f"{dt.date().isoformat()}T{h:02d}:00" for h in hours if h != dt.hour][:4]
        return self._present({"action": "reschedule", "booking_id": booking_id, "provider_id": b["provider_id"],
                              "provider_name": b["provider_name"], "service": b["service"], "slot": b["slot"],
                              "new_slot": new_slot, "price": b["price"], "warnings": [], "alternative_slots": alts})

    def _find_booking(self, booking_id: str) -> dict | None:
        bookings = self.toolbox.call("list_bookings", user_id=self.user_id)["bookings"]
        return next((b for b in bookings if b["booking_id"] == booking_id), None)

    def _present(self, proposal: dict) -> dict:
        s = self.session
        self.approvals.invalidate_session(s.session_id, "superseded_by_new_proposal")
        proposal["shown_turn"] = s.turn
        s.proposal = proposal
        self.store.transition(s, Phase.AWAITING_APPROVAL, f"proposal: {proposal['action']}")
        self.tracer.emit("proposal_shown", proposal={k: proposal.get(k) for k in
                                                     ("action", "provider_id", "slot", "new_slot", "price",
                                                      "booking_id", "hold_id", "warnings")})
        raise EndTurn(self._summary(proposal), {"status": "shown_to_user_for_confirmation"})

    @staticmethod
    def _summary(p: dict) -> str:
        if p["action"] == "book":
            line = (f"Please confirm: book {p['provider_name']} ({p['service']}) on {fmt_slot(p['slot'])} "
                    f"for ₹{p['price']:,}. The slot is held until {datetime.fromisoformat(p['hold_expires_at']):%I:%M %p}.")
        elif p["action"] == "cancel":
            line = (f"Please confirm: cancel booking {p['booking_id']} — {p['provider_name']} ({p['service']}) on "
                    f"{fmt_slot(p['slot'])}.")
        else:
            line = (f"Please confirm: move booking {p['booking_id']} ({p['provider_name']}, {p['service']}) from "
                    f"{fmt_slot(p['slot'])} to {fmt_slot(p['new_slot'])}. Price unchanged at ₹{p['price']:,}.")
        if p.get("warnings"):
            line += " Note: " + "; ".join(p["warnings"]) + "."
        return line + " Reply 'yes' to confirm or 'no' to stop."

    # -- memory -------------------------------------------------------------------
    def _t_remember_preference(self, key: str, value: str, scope: str = "global") -> dict:
        fid = self.memory.add(key, value, source="explicit", scope=scope or "global")
        self.tracer.emit("memory_write", fact_id=fid, key=key, value=value, scope=scope)
        return {"saved": {"fact_id": fid, "key": key, "value": value, "scope": scope or "global"}}

    def _t_forget_preference(self, key: str | None = None, fact_id: int | None = None, scope: str | None = None) -> dict:
        ids = self.memory.forget(key=key, fact_id=fact_id, scope=scope)
        self.tracer.emit("memory_forget", key=key, fact_ids=ids)
        # Anything in the live request that came from the forgotten fact is dropped too.
        if key == "preferred_area" and "area" in self.session.slots.from_memory:
            self.session.slots.area = None
            self.session.slots.from_memory.remove("area")
        return {"forgotten": len(ids)}

    def _t_list_memory(self) -> dict:
        return self.memory.inspect()

    def _t_confirm_inferred(self, fact_id: int, correct: bool) -> dict:
        ok = self.memory.confirm(fact_id) if correct else bool(self.memory.forget(fact_id=fact_id))
        self.tracer.emit("memory_confirm", fact_id=fact_id, correct=correct, ok=ok)
        return {"ok": ok}

    # ============================================================ approval + execution
    def _approve_and_execute(self, user_text: str) -> str:
        s = self.session
        approval_id = self.approvals.grant(s.session_id, s.proposal, user_text)
        self.store.transition(s, Phase.EXECUTING, f"approved {s.proposal['action']}")
        try:
            outcome = self.booking.execute(s, approval_id)
        except ToolError as e:
            self.tracer.emit("error", where="execute", error=str(e))
            outcome = Outcome("unverified", "The provider system errored and I couldn't verify the result. I have "
                                            "not confirmed anything and won't act twice; I'll re-check.")
        return self._apply_outcome(outcome)

    def _apply_outcome(self, o: Outcome) -> str:
        s = self.session
        self.tracer.emit("execution_outcome", status=o.status, result=o.result)
        if o.status in ("confirmed", "already_done"):
            self._verified_this_turn = True
            s.last_result = {**(o.result or {}), "action": s.proposal["action"] if s.proposal else None}
            s.proposal = None
            self.store.transition(s, Phase.DONE, o.status)
            self._snapshot_bookings()
            return o.message
        if o.status == "needs_reapproval":
            s.proposal = o.new_proposal
            s.proposal["shown_turn"] = s.turn
            self.store.transition(s, Phase.AWAITING_APPROVAL, "re-approval required")
            return o.message + "\n\n" + self._summary(s.proposal)
        if o.status == "alternatives":
            s.proposal = None
            self.store.transition(s, Phase.COMPARING, "alternatives offered")
            if o.alternatives:
                alts = "; ".join(f"{a.get('name', a.get('provider_id', ''))} {fmt_slot(a['slot'])}"
                                 + (f" (₹{a['price']:,})" if a.get("price") else "") if isinstance(a, dict)
                                 else fmt_slot(a) for a in o.alternatives)
                return f"{o.message} Alternatives: {alts}. Want me to propose one of these?"
            return o.message + " Want me to look for other options?"
        if o.status == "unverified":
            self.store.transition(s, Phase.VERIFYING, "ambiguous outcome")
            return o.message
        s.proposal = None
        self.store.transition(s, Phase.FAILED, o.message)
        return o.message

    # ================================================================ recovery
    def _recover(self) -> None:
        """Startup reconciliation of in-flight side effects (process restart safety)."""
        for r in self.booking.recover():
            self._apply_recovery(r)

    def _reconcile_in_flight(self) -> str | None:
        rs = self.booking.recover()
        msgs = [self._apply_recovery(r, notify=False) for r in rs]
        msgs = [m for m in msgs if m]
        return "\n".join(msgs) if msgs else None

    def _apply_recovery(self, r: dict, notify: bool = True) -> str | None:
        s, p = self.session, r["proposal"]
        if r["verdict"] == "confirmed":
            self._verified_this_turn = True
            s.last_result = {**(r.get("result") or {}), "action": p["action"]}
            s.proposal = None
            if s.phase != Phase.DONE:
                self.store.transition(s, Phase.DONE, "recovered: confirmed")
            msg = f"Update: I re-checked with the provider and your {p['action']} went through — " \
                  f"{self._describe(s.last_result)} (verified)."
        elif r["verdict"] in ("not_booked", "not_applied"):
            s.proposal = {**p, "shown_turn": s.turn}
            self.store.transition(s, Phase.AWAITING_APPROVAL, "recovered: not applied")
            msg = (f"Update: your earlier {p['action']} request was interrupted and I verified it did NOT go "
                   f"through. Nothing was changed. " + self._summary(s.proposal))
        else:
            if s.phase != Phase.VERIFYING:
                self.store.transition(s, Phase.VERIFYING, "recovered: still unknown")
            msg = "I still can't verify your earlier request with the provider. I won't retry until I can."
        self.store.save(s)
        if notify:
            self.notify(msg, channel="recovery")
            return None
        return msg

    # ============================================================== helpers
    def _snapshot_bookings(self) -> None:
        try:
            for b in self.toolbox.call("list_bookings", user_id=self.user_id)["bookings"]:
                self.db.execute(
                    "INSERT INTO booking_snapshots(booking_id,user_id,status) VALUES(?,?,?) "
                    "ON CONFLICT(booking_id) DO UPDATE SET status=excluded.status", (b["booking_id"], self.user_id,
                                                                                      b["status"]))
        except (ToolError, BudgetExceeded):
            pass

    def _describe(self, r: dict) -> str:
        if not r:
            return "no details"
        parts = [r.get("provider_name") or r.get("provider_id"), fmt_slot(r["slot"]) if r.get("slot") else None,
                 f"₹{r['price']:,}" if r.get("price") else None, f"booking {r.get('booking_id')}"]
        return ", ".join(p for p in parts if p)

    def _safe_status(self) -> str:
        s = self.session
        if s.phase == Phase.AWAITING_APPROVAL and s.proposal:
            return "Nothing has been booked or changed; your pending proposal still needs your confirmation."
        if s.phase == Phase.VERIFYING:
            return "A previous request is still being verified; I won't repeat it."
        return "Nothing has been booked or changed."

    def _guard_claims(self, reply: str) -> str:
        if not self._verified_this_turn and CLAIM_RX.search(reply or ""):
            self.tracer.emit("claim_blocked", original=reply)
            return ("I haven't booked, cancelled or changed anything. " + self._safe_status()
                    + (" " + self._summary(self.session.proposal) if self.session.proposal and
                       self.session.phase == Phase.AWAITING_APPROVAL else ""))
        return reply

    def _save_message(self, role: str, content: str) -> None:
        self.db.execute("INSERT INTO messages(session_id,user_id,role,content,created_at) VALUES(?,?,?,?,?)",
                        (self.session_id, self.user_id, role, content, self.now().isoformat()))


def _tool_result(tool_use_id: str, out: dict, is_error: bool) -> dict:
    block = {"type": "tool_result", "tool_use_id": tool_use_id,
             "content": json.dumps(out, ensure_ascii=False, default=str)}
    if is_error:
        block["is_error"] = True
    return block
