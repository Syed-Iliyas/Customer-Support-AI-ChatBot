"""SQLite persistence: conversation state, interaction logs and demo orders."""
from __future__ import annotations

import json
import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone

DEFAULT_DB = os.environ.get(
    "CHATBOT_DB", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "chatbot.db")
)

SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    session_id TEXT PRIMARY KEY,
    state      TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS chat_logs (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id   TEXT NOT NULL,
    user_message TEXT NOT NULL,
    bot_response TEXT NOT NULL,
    intent       TEXT,
    confidence   REAL,
    is_fallback  INTEGER NOT NULL DEFAULT 0,
    latency_ms   REAL,
    feedback     INTEGER,
    created_at   TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_logs_session ON chat_logs(session_id);
CREATE INDEX IF NOT EXISTS idx_logs_created ON chat_logs(created_at);
CREATE TABLE IF NOT EXISTS orders (
    order_id TEXT PRIMARY KEY,
    item     TEXT NOT NULL,
    status   TEXT NOT NULL,
    carrier  TEXT,
    eta      TEXT
);
"""

DEMO_ORDERS = [
    ("10001", "Wireless Headphones", "Shipped", "BlueDart", "2 days"),
    ("10002", "Running Shoes", "Processing", None, "5-7 days"),
    ("10003", "Laptop Backpack", "Delivered", "Delhivery", None),
    ("10004", "Smart Watch", "Out for delivery", "Ecom Express", "today"),
    ("10005", "Desk Lamp", "Cancelled", None, None),
]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class Database:
    def __init__(self, path: str | None = None):
        self.path = path or DEFAULT_DB
        self.init()

    @contextmanager
    def conn(self):
        c = sqlite3.connect(self.path, timeout=10)
        c.row_factory = sqlite3.Row
        try:
            yield c
            c.commit()
        finally:
            c.close()

    def init(self) -> None:
        with self.conn() as c:
            c.execute("PRAGMA journal_mode=WAL")  # readers don't block the writer; safe with several workers
            c.executescript(SCHEMA)
            if c.execute("SELECT COUNT(*) FROM orders").fetchone()[0] == 0:
                c.executemany("INSERT INTO orders VALUES (?,?,?,?,?)", DEMO_ORDERS)

    # ---- sessions (conversation context) -------------------------------
    @staticmethod
    def _blank_state() -> dict:
        return {"last_intent": None, "pending": None, "entities": {}, "fallback_streak": 0, "turns": 0}

    def get_session(self, session_id: str) -> dict:
        with self.conn() as c:
            row = c.execute("SELECT state FROM sessions WHERE session_id=?", (session_id,)).fetchone()
        if not row:
            return self._blank_state()
        state = self._blank_state()
        state.update(json.loads(row["state"]))
        return state

    def save_session(self, session_id: str, state: dict) -> None:
        now = _now()
        with self.conn() as c:
            c.execute(
                """INSERT INTO sessions(session_id, state, created_at, updated_at) VALUES (?,?,?,?)
                   ON CONFLICT(session_id) DO UPDATE SET state=excluded.state, updated_at=excluded.updated_at""",
                (session_id, json.dumps(state), now, now),
            )

    # ---- logs -----------------------------------------------------------
    def log_chat(self, session_id, user_message, bot_response, intent, confidence, is_fallback, latency_ms) -> int:
        with self.conn() as c:
            cur = c.execute(
                """INSERT INTO chat_logs(session_id,user_message,bot_response,intent,confidence,is_fallback,latency_ms,created_at)
                   VALUES (?,?,?,?,?,?,?,?)""",
                (session_id, user_message, bot_response, intent, confidence, int(is_fallback), latency_ms, _now()),
            )
            return cur.lastrowid

    def set_feedback(self, log_id: int, value: int) -> bool:
        with self.conn() as c:
            cur = c.execute("UPDATE chat_logs SET feedback=? WHERE id=?", (value, log_id))
            return cur.rowcount > 0

    def recent_logs(self, limit: int = 100, offset: int = 0, session_id: str | None = None) -> list[dict]:
        q = "SELECT * FROM chat_logs"
        args: list = []
        if session_id:
            q += " WHERE session_id=?"
            args.append(session_id)
        q += " ORDER BY id DESC LIMIT ? OFFSET ?"
        args += [limit, offset]
        with self.conn() as c:
            return [dict(r) for r in c.execute(q, args).fetchall()]

    def stats(self) -> dict:
        with self.conn() as c:
            total = c.execute("SELECT COUNT(*) FROM chat_logs").fetchone()[0]
            sessions = c.execute("SELECT COUNT(DISTINCT session_id) FROM chat_logs").fetchone()[0]
            fallbacks = c.execute("SELECT COUNT(*) FROM chat_logs WHERE is_fallback=1").fetchone()[0]
            up = c.execute("SELECT COUNT(*) FROM chat_logs WHERE feedback=1").fetchone()[0]
            down = c.execute("SELECT COUNT(*) FROM chat_logs WHERE feedback=-1").fetchone()[0]
            avg_lat = c.execute("SELECT AVG(latency_ms) FROM chat_logs").fetchone()[0]
            top = c.execute(
                """SELECT intent, COUNT(*) n FROM chat_logs WHERE intent IS NOT NULL
                   GROUP BY intent ORDER BY n DESC LIMIT 8"""
            ).fetchall()
            unanswered = c.execute(
                """SELECT user_message, COUNT(*) n FROM chat_logs WHERE is_fallback=1
                   GROUP BY user_message ORDER BY n DESC, MAX(id) DESC LIMIT 8"""
            ).fetchall()
        return {
            "total_messages": total,
            "sessions": sessions,
            "fallback_rate": round(fallbacks / total, 3) if total else 0.0,
            "thumbs_up": up,
            "thumbs_down": down,
            "avg_latency_ms": round(avg_lat, 1) if avg_lat else 0.0,
            "top_intents": [{"intent": r["intent"], "count": r["n"]} for r in top],
            "unanswered": [{"message": r["user_message"], "count": r["n"]} for r in unanswered],
        }

    # ---- orders (demo data) ---------------------------------------------
    def get_order(self, order_id: str) -> dict | None:
        with self.conn() as c:
            row = c.execute("SELECT * FROM orders WHERE order_id=?", (order_id,)).fetchone()
        return dict(row) if row else None

    def set_order_status(self, order_id: str, status: str) -> None:
        with self.conn() as c:
            c.execute("UPDATE orders SET status=? WHERE order_id=?", (status, order_id))
