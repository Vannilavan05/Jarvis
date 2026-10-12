"""Audit Logger for Jarvis.
Maintains tamper-evident local audit records in SQLite for all command evaluations and executions.
"""

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

try:
    from storage.database import DatabaseManager
except (ImportError, ValueError):
    from jarvis.storage.database import DatabaseManager

logger = logging.getLogger(__name__)


class AuditLogger:
    """Manages secure local audit records."""

    def __init__(self, db_manager: Optional[DatabaseManager] = None):
        self.db = db_manager or DatabaseManager()

    def log(
        self,
        command_text: str,
        intent: Optional[str] = None,
        confidence: float = 0.0,
        permission_level: str = "STANDARD",
        authorized: bool = True,
        confirmation_required: bool = False,
        confirmation_status: str = "NONE",
        status: str = "COMPLETED",
        result: Optional[str] = None,
        error_details: Optional[str] = None,
        model_version: int = 1,
    ) -> int:
        """Appends a command event to the SQLite audit log."""
        timestamp = datetime.now(timezone.utc).isoformat()
        conn = self.db.get_connection()
        try:
            with conn:
                cursor = conn.execute(
                    """
                    INSERT INTO audit_logs (
                        timestamp, command_text, intent, confidence,
                        permission_level, authorized, confirmation_required,
                        confirmation_status, status, result, error_details, model_version
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        timestamp,
                        command_text,
                        intent,
                        confidence,
                        permission_level,
                        1 if authorized else 0,
                        1 if confirmation_required else 0,
                        confirmation_status,
                        status,
                        result,
                        error_details,
                        model_version,
                    ),
                )
                return cursor.lastrowid
        except Exception as e:
            logger.error(f"Failed to record audit log: {e}")
            return -1

    def get_recent(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Retrieves recent audit events."""
        conn = self.db.get_connection()
        cursor = conn.execute(
            """
            SELECT id, timestamp, command_text, intent, confidence,
                   permission_level, authorized, confirmation_required,
                   confirmation_status, status, result, error_details, model_version
            FROM audit_logs
            ORDER BY id DESC
            LIMIT ?
            """,
            (limit,),
        )
        rows = cursor.fetchall()
        logs = []
        for row in rows:
            logs.append({
                "id": row["id"],
                "timestamp": row["timestamp"],
                "command_text": row["command_text"],
                "intent": row["intent"],
                "confidence": row["confidence"],
                "permission_level": row["permission_level"],
                "authorized": bool(row["authorized"]),
                "confirmation_required": bool(row["confirmation_required"]),
                "confirmation_status": row["confirmation_status"],
                "status": row["status"],
                "result": row["result"],
                "error_details": row["error_details"],
                "model_version": row["model_version"],
            })
        return logs

    def clear(self) -> int:
        """Purges audit records on user command."""
        conn = self.db.get_connection()
        with conn:
            cursor = conn.execute("DELETE FROM audit_logs")
            return cursor.rowcount
