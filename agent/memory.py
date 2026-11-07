"""Persistent user memory. Facts are explicit (user said it) or inferred (derived from history).
Inferred facts never reach the model until the user confirms them. Forgetting is a tombstone
(forgotten_at), so forgotten facts are provably excluded rather than silently re-learned."""
from __future__ import annotations

from datetime import datetime

# Facts that matter for any booking, regardless of service.
BOOKING_KEYS = {"preferred_area", "earliest_hour", "latest_hour", "access_notes", "pets", "contact_preference",
                "budget_max", "preferred_provider", "payment_preference", "floor_access"}


class Memory:
    def __init__(self, db, user_id: str, now_fn=datetime.now):
        self.db, self.user_id, self.now = db, user_id, now_fn

    def _ts(self) -> str:
        return self.now().isoformat(timespec="seconds")

    # ----------------------------------------------------------------- writes
    def add(self, key: str, value, source: str = "explicit", scope: str = "global",
            confirmed: bool | None = None) -> int:
        """Upsert by (key, scope). An explicit statement supersedes any prior value (a correction)."""
        confirmed = (source == "explicit") if confirmed is None else confirmed
        existing = self.db.one(
            "SELECT id FROM facts WHERE user_id=? AND key=? AND scope=? AND forgotten_at IS NULL",
            (self.user_id, key, scope))
        if existing:
            self.db.execute("UPDATE facts SET value=?, source=?, confirmed=?, updated_at=? WHERE id=?",
                            (str(value), source, int(confirmed), self._ts(), existing["id"]))
            return existing["id"]
        cur = self.db.execute(
            "INSERT INTO facts(user_id,key,value,scope,source,confirmed,created_at,updated_at) VALUES(?,?,?,?,?,?,?,?)",
            (self.user_id, key, str(value), scope, source, int(confirmed), self._ts(), self._ts()))
        return cur.lastrowid

    def forget(self, key: str | None = None, fact_id: int | None = None, scope: str | None = None) -> list[int]:
        """Tombstone matching active facts. Returns the ids forgotten."""
        q = "SELECT id FROM facts WHERE user_id=? AND forgotten_at IS NULL"
        params: list = [self.user_id]
        if fact_id is not None:
            q += " AND id=?"
            params.append(fact_id)
        elif key is not None:
            q += " AND key=?"
            params.append(key)
            if scope:
                q += " AND scope=?"
                params.append(scope)
        else:
            return []
        ids = [r["id"] for r in self.db.all(q, tuple(params))]
        for i in ids:
            self.db.execute("UPDATE facts SET forgotten_at=? WHERE id=?", (self._ts(), i))
        return ids

    def confirm(self, fact_id: int) -> bool:
        return self.db.execute(
            "UPDATE facts SET confirmed=1, updated_at=? WHERE id=? AND user_id=? AND forgotten_at IS NULL",
            (self._ts(), fact_id, self.user_id)).rowcount == 1

    # ------------------------------------------------------------------ reads
    def active(self) -> list[dict]:
        return self.db.all("SELECT * FROM facts WHERE user_id=? AND forgotten_at IS NULL ORDER BY id",
                           (self.user_id,))

    def eligible(self) -> list[dict]:
        """The only facts allowed to reach the model or drive behaviour."""
        return self.db.all(
            "SELECT * FROM facts WHERE user_id=? AND forgotten_at IS NULL AND (source='explicit' OR confirmed=1) "
            "ORDER BY updated_at DESC, id DESC", (self.user_id,))

    def get(self, key: str, scope: str = "global"):
        for f in self.eligible():
            if f["key"] == key and f["scope"] == scope:
                return f["value"]
        return None

    def inspect(self) -> dict:
        out = {"explicit": [], "inferred_confirmed": [], "inferred_unconfirmed": []}
        for f in self.active():
            item = {"id": f["id"], "key": f["key"], "value": f["value"], "scope": f["scope"]}
            if f["source"] == "explicit":
                out["explicit"].append(item)
            elif f["confirmed"]:
                out["inferred_confirmed"].append(item)
            else:
                out["inferred_unconfirmed"].append(item)
        return out

    def relevant(self, service: str | None) -> list[dict]:
        """Eligible facts ranked by relevance to the current request. Facts scoped to a *different*
        service are excluded entirely."""
        scored = []
        for f in self.eligible():
            if f["scope"] not in ("global", service):
                continue
            score = 0
            if service and f["scope"] == service:
                score += 3
            if f["key"] in BOOKING_KEYS or f["key"].startswith(("prefer", "budget", "earliest", "latest")):
                score += 2
            scored.append((score, f["updated_at"] or "", f))
        scored.sort(key=lambda t: (t[0], t[1]), reverse=True)
        return [f for _, _, f in scored]
