"""SQLite Database Manager for Jarvis.
Provides thread-safe connections and schema initialization.
"""

import sqlite3
import threading
from pathlib import Path
from typing import Optional, Union

try:
    from config import DATABASE_PATH, DATA_DIR
except (ImportError, ValueError):
    from jarvis.config import DATABASE_PATH, DATA_DIR


class DatabaseManager:
    """Manages the SQLite database connection and table schema for Jarvis."""

    _instance = None
    _lock = threading.Lock()

    def __init__(self, db_path: Optional[Union[str, Path]] = None):
        self.db_path = Path(db_path) if db_path else DATABASE_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
        self.init_schema()

    def get_connection(self) -> sqlite3.Connection:
        """Returns a thread-local SQLite connection with Row row_factory."""
        if not hasattr(self._local, "connection") or self._local.connection is None:
            conn = sqlite3.connect(str(self.db_path), timeout=20.0)
            conn.row_factory = sqlite3.Row
            # Enable WAL mode for better concurrency
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.execute("PRAGMA foreign_keys=ON;")
            self._local.connection = conn
        return self._local.connection

    def close(self):
        """Closes the thread-local connection if open."""
        if hasattr(self._local, "connection") and self._local.connection is not None:
            try:
                self._local.connection.close()
            except Exception:
                pass
            self._local.connection = None

    def init_schema(self):
        """Initializes tables for training examples, audit logs, custom commands, and checkpoints."""
        conn = self.get_connection()
        with conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS training_examples (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    text TEXT NOT NULL UNIQUE COLLATE NOCASE,
                    intent TEXT NOT NULL,
                    entities TEXT DEFAULT '{}',
                    approved INTEGER DEFAULT 1,
                    created_at TEXT NOT NULL,
                    dataset_version INTEGER DEFAULT 1
                );
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS audit_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    command_text TEXT NOT NULL,
                    intent TEXT,
                    confidence REAL DEFAULT 0.0,
                    permission_level TEXT DEFAULT 'STANDARD',
                    authorized INTEGER NOT NULL,
                    confirmation_required INTEGER DEFAULT 0,
                    confirmation_status TEXT DEFAULT 'NONE',
                    status TEXT NOT NULL,
                    result TEXT,
                    error_details TEXT,
                    model_version INTEGER DEFAULT 1
                );
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS custom_commands (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    phrase TEXT UNIQUE NOT NULL COLLATE NOCASE,
                    intent TEXT NOT NULL,
                    parameters TEXT DEFAULT '{}',
                    created_at TEXT NOT NULL
                );
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS model_checkpoints (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    version INTEGER UNIQUE NOT NULL,
                    file_path TEXT NOT NULL,
                    vocab_path TEXT NOT NULL,
                    dataset_size INTEGER DEFAULT 0,
                    accuracy REAL DEFAULT 0.0,
                    loss REAL DEFAULT 0.0,
                    created_at TEXT NOT NULL,
                    is_active INTEGER DEFAULT 0
                );
                """
            )
