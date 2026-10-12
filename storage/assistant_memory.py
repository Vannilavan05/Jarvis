"""Persistent local memory and automation progress for Jarvis.

All records remain in the existing local SQLite database. This module stores
conversation context, explicit user memories, and task/step status; it does not
execute commands or grant permissions.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional


_SCHEMA = (
    """CREATE TABLE IF NOT EXISTS assistant_messages (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id TEXT NOT NULL,
        role TEXT NOT NULL CHECK(role IN ('user','assistant','system')),
        content TEXT NOT NULL,
        created_at TEXT NOT NULL
    )""",
    """CREATE TABLE IF NOT EXISTS assistant_memories (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        memory_key TEXT NOT NULL UNIQUE COLLATE NOCASE,
        memory_value TEXT NOT NULL,
        updated_at TEXT NOT NULL
    )""",
    """CREATE TABLE IF NOT EXISTS automation_tasks (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        title TEXT NOT NULL,
        status TEXT NOT NULL CHECK(status IN ('planned','running','waiting_confirmation','completed','failed','cancelled')),
        current_step INTEGER NOT NULL DEFAULT 0,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        error TEXT
    )""",
    """CREATE TABLE IF NOT EXISTS automation_steps (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        task_id INTEGER NOT NULL REFERENCES automation_tasks(id) ON DELETE CASCADE,
        step_index INTEGER NOT NULL,
        intent TEXT NOT NULL,
        parameters_json TEXT NOT NULL DEFAULT '{}',
        status TEXT NOT NULL CHECK(status IN ('pending','running','waiting_confirmation','completed','failed','skipped')),
        result TEXT,
        updated_at TEXT NOT NULL,
        UNIQUE(task_id, step_index)
    )""",
    "CREATE INDEX IF NOT EXISTS idx_assistant_messages_session ON assistant_messages(session_id, id)",
    "CREATE INDEX IF NOT EXISTS idx_automation_tasks_status ON automation_tasks(status, updated_at)",
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class AssistantMemory:
    """Store and retrieve user-approved memories and resumable task progress."""

    def __init__(self, db_manager):
        self.db = db_manager
        conn = self.db.get_connection()
        with conn:
            for statement in _SCHEMA:
                conn.execute(statement)

    def save_message(self, session_id: str, role: str, content: str) -> int:
        if role not in {"user", "assistant", "system"}:
            raise ValueError("Unsupported message role.")
        if not isinstance(session_id, str) or not session_id.strip():
            raise ValueError("session_id must not be empty.")
        if not isinstance(content, str) or not content.strip() or len(content) > 12000:
            raise ValueError("Message must contain 1 to 12000 characters.")
        conn = self.db.get_connection()
        with conn:
            cur = conn.execute(
                "INSERT INTO assistant_messages(session_id, role, content, created_at) VALUES(?,?,?,?)",
                (session_id.strip(), role, content.strip(), _now()),
            )
        return int(cur.lastrowid)

    def recent_messages(self, session_id: str, limit: int = 12) -> List[Dict[str, Any]]:
        limit = max(1, min(int(limit), 100))
        rows = self.db.get_connection().execute(
            "SELECT role, content, created_at FROM assistant_messages WHERE session_id=? ORDER BY id DESC LIMIT ?",
            (session_id, limit),
        ).fetchall()
        return [dict(row) for row in reversed(rows)]

    def remember(self, key: str, value: str) -> None:
        key, value = str(key).strip(), str(value).strip()
        if not key or len(key) > 160 or not value or len(value) > 4000:
            raise ValueError("Memory key/value is empty or exceeds the allowed length.")
        conn = self.db.get_connection()
        with conn:
            conn.execute(
                "INSERT INTO assistant_memories(memory_key,memory_value,updated_at) VALUES(?,?,?) "
                "ON CONFLICT(memory_key) DO UPDATE SET memory_value=excluded.memory_value, updated_at=excluded.updated_at",
                (key, value, _now()),
            )

    def recall(self, key: Optional[str] = None, limit: int = 30) -> List[Dict[str, str]]:
        conn = self.db.get_connection()
        if key:
            rows = conn.execute(
                "SELECT memory_key,memory_value,updated_at FROM assistant_memories WHERE memory_key=? COLLATE NOCASE",
                (key.strip(),),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT memory_key,memory_value,updated_at FROM assistant_memories ORDER BY updated_at DESC LIMIT ?",
                (max(1, min(int(limit), 100)),),
            ).fetchall()
        return [dict(row) for row in rows]

    def create_task(self, title: str, steps: Iterable[Dict[str, Any]]) -> int:
        title = str(title).strip()
        steps = list(steps)
        if not title or len(title) > 240:
            raise ValueError("Task title must contain 1 to 240 characters.")
        if not steps or len(steps) > 20:
            raise ValueError("Automation tasks must contain between 1 and 20 steps.")
        for step in steps:
            if not isinstance(step, dict) or not isinstance(step.get("intent"), str) or not isinstance(step.get("parameters", {}), dict):
                raise ValueError("Each step must contain an intent and an object of parameters.")
        conn = self.db.get_connection()
        now = _now()
        with conn:
            cur = conn.execute(
                "INSERT INTO automation_tasks(title,status,current_step,created_at,updated_at) VALUES(?,?,?,?,?)",
                (title, "planned", 0, now, now),
            )
            task_id = int(cur.lastrowid)
            conn.executemany(
                "INSERT INTO automation_steps(task_id,step_index,intent,parameters_json,status,updated_at) VALUES(?,?,?,?,?,?)",
                [(task_id, i, s["intent"], json.dumps(s.get("parameters", {}), ensure_ascii=False), "pending", now) for i, s in enumerate(steps)],
            )
        return task_id

    def get_task(self, task_id: int) -> Optional[Dict[str, Any]]:
        conn = self.db.get_connection()
        task = conn.execute("SELECT * FROM automation_tasks WHERE id=?", (int(task_id),)).fetchone()
        if task is None:
            return None
        result = dict(task)
        rows = conn.execute("SELECT * FROM automation_steps WHERE task_id=? ORDER BY step_index", (int(task_id),)).fetchall()
        result["steps"] = [dict(row, parameters=json.loads(row["parameters_json"])) for row in rows]
        return result

    def update_task(self, task_id: int, status: str, current_step: Optional[int] = None, error: Optional[str] = None) -> None:
        allowed = {"planned", "running", "waiting_confirmation", "completed", "failed", "cancelled"}
        if status not in allowed:
            raise ValueError("Unsupported task status.")
        conn = self.db.get_connection()
        with conn:
            if current_step is None:
                conn.execute("UPDATE automation_tasks SET status=?,updated_at=?,error=? WHERE id=?", (status, _now(), error, int(task_id)))
            else:
                conn.execute("UPDATE automation_tasks SET status=?,current_step=?,updated_at=?,error=? WHERE id=?", (status, int(current_step), _now(), error, int(task_id)))

    def update_step(self, task_id: int, step_index: int, status: str, result: Optional[str] = None) -> None:
        allowed = {"pending", "running", "waiting_confirmation", "completed", "failed", "skipped"}
        if status not in allowed:
            raise ValueError("Unsupported step status.")
        conn = self.db.get_connection()
        with conn:
            conn.execute(
                "UPDATE automation_steps SET status=?,result=?,updated_at=? WHERE task_id=? AND step_index=?",
                (status, result[:2000] if result else None, _now(), int(task_id), int(step_index)),
            )

    def recent_tasks(self, limit: int = 10) -> List[Dict[str, Any]]:
        rows = self.db.get_connection().execute(
            "SELECT id,title,status,current_step,created_at,updated_at,error FROM automation_tasks ORDER BY id DESC LIMIT ?",
            (max(1, min(int(limit), 50)),),
        ).fetchall()
        return [dict(row) for row in rows]

    def pending_tasks(self) -> List[Dict[str, Any]]:
        rows = self.db.get_connection().execute(
            "SELECT id,title,status,current_step,created_at,updated_at,error FROM automation_tasks "
            "WHERE status IN ('planned','running','waiting_confirmation','failed') ORDER BY id DESC"
        ).fetchall()
        return [dict(row) for row in rows]
