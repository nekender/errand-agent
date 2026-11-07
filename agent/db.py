"""SQLite persistence for agent-side state. Survives process restarts."""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    session_id TEXT PRIMARY KEY, user_id TEXT NOT NULL, state_json TEXT NOT NULL, updated_at TEXT
);
CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT, session_id TEXT, user_id TEXT, role TEXT,
    content TEXT, created_at TEXT
);
CREATE TABLE IF NOT EXISTS facts (
    id INTEGER PRIMARY KEY AUTOINCREMENT, user_id TEXT NOT NULL, key TEXT NOT NULL, value TEXT NOT NULL,
    scope TEXT NOT NULL DEFAULT 'global',          -- 'global' or a service name
    source TEXT NOT NULL CHECK (source IN ('explicit','inferred')),
    confirmed INTEGER NOT NULL DEFAULT 0,
    created_at TEXT, updated_at TEXT, forgotten_at TEXT
);
CREATE TABLE IF NOT EXISTS approvals (
    approval_id TEXT PRIMARY KEY, session_id TEXT, action TEXT, fingerprint TEXT,
    payload_json TEXT, status TEXT, created_at TEXT, consumed_at TEXT
);
CREATE TABLE IF NOT EXISTS pending_ops (
    idem_key TEXT PRIMARY KEY, session_id TEXT, user_id TEXT, action TEXT, op_tuple TEXT,
    approval_id TEXT, payload_json TEXT, status TEXT, result_json TEXT,
    attempts INTEGER DEFAULT 0, created_at TEXT, updated_at TEXT
);
CREATE TABLE IF NOT EXISTS context_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT, session_id TEXT, turn INTEGER, model_call INTEGER,
    fact_ids TEXT, message_ids TEXT, sections_json TEXT, total_tokens INTEGER, budget INTEGER, created_at TEXT
);
CREATE TABLE IF NOT EXISTS booking_snapshots (
    booking_id TEXT PRIMARY KEY, user_id TEXT, status TEXT, handled_event TEXT
);
CREATE TABLE IF NOT EXISTS outbox (
    id INTEGER PRIMARY KEY AUTOINCREMENT, session_id TEXT, content TEXT, delivered INTEGER DEFAULT 0, created_at TEXT
);
"""


class DB:
    def __init__(self, path: str | Path = ":memory:"):
        self.path = str(path)
        self.conn = sqlite3.connect(self.path, check_same_thread=False, isolation_level=None)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA)

    def execute(self, sql: str, params: tuple | dict = ()) -> sqlite3.Cursor:
        return self.conn.execute(sql, params)

    def one(self, sql: str, params: tuple | dict = ()) -> dict | None:
        row = self.conn.execute(sql, params).fetchone()
        return dict(row) if row else None

    def all(self, sql: str, params: tuple | dict = ()) -> list[dict]:
        return [dict(r) for r in self.conn.execute(sql, params).fetchall()]

    @staticmethod
    def dumps(obj) -> str:
        return json.dumps(obj, sort_keys=True, default=str)
