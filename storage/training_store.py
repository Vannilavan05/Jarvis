"""Training Store for Jarvis.
Manages storage, validation, versioning, and retrieval of approved training examples.
"""

import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from .database import DatabaseManager

logger = logging.getLogger(__name__)


class TrainingStore:
    """Manages the training dataset in SQLite with schema validation and versioning."""

    VALID_INTENTS = {
        "open_application",
        "close_application",
        "web_search",
        "read_file",
        "create_file",
        "delete_file",
        "list_files",
        "system_info",
        "system_control",
        "start_training",
        "add_example",
        "review_examples",
        "train_model",
        "evaluate_model",
        "get_model_version",
        "rollback_model",
        "stop_listening",
        "pause_listening",
        "resume_listening",
        "cancel_action",
        "custom_command",
    }

    def __init__(self, db_manager: Optional[DatabaseManager] = None):
        self.db = db_manager or DatabaseManager()

    def add_example(
        self,
        text: str,
        intent: str,
        entities: Optional[Dict[str, Any]] = None,
        approved: bool = True,
        dataset_version: Optional[int] = None,
    ) -> Tuple[bool, str, Optional[int]]:
        """Validates and inserts a training example.

        Returns (success, message, record_id).
        """
        cleaned_text = text.strip()
        if not cleaned_text or len(cleaned_text) < 2:
            return False, "Training text is too short or empty.", None

        if intent not in self.VALID_INTENTS:
            return False, f"Unsupported intent '{intent}'. Must be registered.", None

        entities_dict = entities or {}
        entities_json = json.dumps(entities_dict)
        now_iso = datetime.now(timezone.utc).isoformat()
        version = dataset_version or self.get_dataset_version()

        conn = self.db.get_connection()
        try:
            with conn:
                cursor = conn.execute(
                    """
                    INSERT INTO training_examples (text, intent, entities, approved, created_at, dataset_version)
                    VALUES (?, ?, ?, ?, ?, ?)
                    ON CONFLICT(text) DO UPDATE SET
                        intent = excluded.intent,
                        entities = excluded.entities,
                        approved = excluded.approved,
                        dataset_version = excluded.dataset_version;
                    """,
                    (cleaned_text, intent, entities_json, 1 if approved else 0, now_iso, version),
                )
                record_id = cursor.lastrowid
                return True, f"Example '{cleaned_text}' registered for intent '{intent}'.", record_id
        except Exception as e:
            logger.error(f"Failed to add training example: {e}")
            return False, f"Database error: {str(e)}", None

    def get_all_examples(self, approved_only: bool = True) -> List[Dict[str, Any]]:
        """Retrieves all dataset records."""
        conn = self.db.get_connection()
        query = "SELECT * FROM training_examples"
        params: List[Any] = []
        if approved_only:
            query += " WHERE approved = 1"
        query += " ORDER BY id ASC"

        cursor = conn.execute(query, params)
        rows = cursor.fetchall()
        results = []
        for row in rows:
            results.append({
                "id": row["id"],
                "text": row["text"],
                "intent": row["intent"],
                "entities": json.loads(row["entities"] or "{}"),
                "approved": bool(row["approved"]),
                "created_at": row["created_at"],
                "dataset_version": row["dataset_version"],
            })
        return results

    def get_last_example(self) -> Optional[Dict[str, Any]]:
        """Retrieves the most recently added training example."""
        conn = self.db.get_connection()
        cursor = conn.execute("SELECT * FROM training_examples ORDER BY id DESC LIMIT 1")
        row = cursor.fetchone()
        if not row:
            return None
        return {
            "id": row["id"],
            "text": row["text"],
            "intent": row["intent"],
            "entities": json.loads(row["entities"] or "{}"),
            "approved": bool(row["approved"]),
            "created_at": row["created_at"],
            "dataset_version": row["dataset_version"],
        }

    def update_example(
        self,
        example_id: int,
        text: Optional[str] = None,
        intent: Optional[str] = None,
        entities: Optional[Dict[str, Any]] = None,
        approved: Optional[bool] = None,
    ) -> Tuple[bool, str]:
        """Updates fields of an existing training example."""
        conn = self.db.get_connection()
        fields = []
        params: List[Any] = []

        if text is not None:
            fields.append("text = ?")
            params.append(text.strip())
        if intent is not None:
            if intent not in self.VALID_INTENTS:
                return False, f"Invalid intent '{intent}'."
            fields.append("intent = ?")
            params.append(intent)
        if entities is not None:
            fields.append("entities = ?")
            params.append(json.dumps(entities))
        if approved is not None:
            fields.append("approved = ?")
            params.append(1 if approved else 0)

        if not fields:
            return False, "No fields specified for update."

        params.append(example_id)
        query = f"UPDATE training_examples SET {', '.join(fields)} WHERE id = ?"

        try:
            with conn:
                cursor = conn.execute(query, params)
                if cursor.rowcount == 0:
                    return False, f"Example ID {example_id} not found."
                return True, f"Example ID {example_id} updated successfully."
        except Exception as e:
            return False, f"Database error: {str(e)}"

    def delete_example(self, example_id: int) -> Tuple[bool, str]:
        """Deletes a training example by ID."""
        conn = self.db.get_connection()
        try:
            with conn:
                cursor = conn.execute("DELETE FROM training_examples WHERE id = ?", (example_id,))
                if cursor.rowcount == 0:
                    return False, f"Example ID {example_id} not found."
                return True, f"Example ID {example_id} deleted."
        except Exception as e:
            return False, f"Failed to delete example: {str(e)}"

    def delete_last_example(self) -> Tuple[bool, str]:
        """Deletes the most recently added training example."""
        last = self.get_last_example()
        if not last:
            return False, "No training examples exist to delete."
        return self.delete_example(last["id"])

    def count_examples(self, approved_only: bool = True) -> int:
        """Counts total training examples."""
        conn = self.db.get_connection()
        query = "SELECT COUNT(*) FROM training_examples"
        if approved_only:
            query += " WHERE approved = 1"
        cursor = conn.execute(query)
        return cursor.fetchone()[0]

    def get_dataset_version(self) -> int:
        """Returns the highest dataset version present, minimum 1."""
        conn = self.db.get_connection()
        cursor = conn.execute("SELECT MAX(dataset_version) FROM training_examples")
        val = cursor.fetchone()[0]
        return val if val is not None else 1

    def bump_dataset_version(self) -> int:
        """Increments the current dataset version for tracking."""
        new_version = self.get_dataset_version() + 1
        return new_version

    def seed_default_examples(self) -> int:
        """Populates comprehensive baseline training examples if the dataset is empty."""
        if self.count_examples(approved_only=False) > 0:
            return 0

        baseline_dataset = [
            # Application launcher
            ("open chrome", "open_application", {"application": "chrome"}),
            ("launch google chrome", "open_application", {"application": "chrome"}),
            ("launch browser", "open_application", {"application": "chrome"}),
            ("open my browser", "open_application", {"application": "chrome"}),
            ("open notepad", "open_application", {"application": "notepad"}),
            ("start text editor", "open_application", {"application": "notepad"}),
            ("open calculator", "open_application", {"application": "calculator"}),
            ("launch calculator", "open_application", {"application": "calculator"}),
            ("open file explorer", "open_application", {"application": "explorer"}),
            ("open vs code", "open_application", {"application": "code"}),
            ("close chrome", "close_application", {"application": "chrome"}),
            ("terminate notepad", "close_application", {"application": "notepad"}),
            ("close calculator", "close_application", {"application": "calculator"}),
            # Web search
            ("search the web for quantum physics", "web_search", {"query": "quantum physics"}),
            ("google artificial intelligence breakthroughs", "web_search", {"query": "artificial intelligence breakthroughs"}),
            ("search online for latest space discoveries", "web_search", {"query": "latest space discoveries"}),
            ("look up python tutorials", "web_search", {"query": "python tutorials"}),
            ("search for weather forecast today", "web_search", {"query": "weather forecast today"}),
            ("web search machine learning papers", "web_search", {"query": "machine learning papers"}),
            # File operations
            ("read file notes.txt", "read_file", {"filename": "notes.txt"}),
            ("show content of document.txt", "read_file", {"filename": "document.txt"}),
            ("create file summary.txt", "create_file", {"filename": "summary.txt"}),
            ("make a new file called test.txt", "create_file", {"filename": "test.txt"}),
            ("delete file temporary.txt", "delete_file", {"filename": "temporary.txt"}),
            ("remove old_log.txt", "delete_file", {"filename": "old_log.txt"}),
            ("list files in directory", "list_files", {}),
            ("show files", "list_files", {}),
            ("what files are here", "list_files", {}),
            # System info
            ("how is the cpu usage", "system_info", {"metric": "cpu"}),
            ("check cpu performance", "system_info", {"metric": "cpu"}),
            ("what is my ram usage", "system_info", {"metric": "memory"}),
            ("check memory status", "system_info", {"metric": "memory"}),
            ("how much disk space is left", "system_info", {"metric": "disk"}),
            ("check battery percentage", "system_info", {"metric": "battery"}),
            ("tell me system specs", "system_info", {"metric": "all"}),
            ("show system status", "system_info", {"metric": "all"}),
            # System controls
            ("increase volume", "system_control", {"action": "volume_up"}),
            ("turn up the sound", "system_control", {"action": "volume_up"}),
            ("decrease volume", "system_control", {"action": "volume_down"}),
            ("turn down the sound", "system_control", {"action": "volume_down"}),
            ("mute audio", "system_control", {"action": "mute"}),
            ("unmute audio", "system_control", {"action": "unmute"}),
            ("lock the screen", "system_control", {"action": "lock"}),
            # Training commands
            ("enter training mode", "start_training", {}),
            ("start training mode", "start_training", {}),
            ("add a training example", "add_example", {}),
            ("review my training examples", "review_examples", {}),
            ("show training examples", "review_examples", {}),
            ("train the model", "train_model", {}),
            ("retrain neural network", "train_model", {}),
            ("evaluate the model", "evaluate_model", {}),
            ("check model accuracy", "evaluate_model", {}),
            ("tell me the current model version", "get_model_version", {}),
            ("what is the model version", "get_model_version", {}),
            ("restore the previous model", "rollback_model", {}),
            ("rollback model", "rollback_model", {}),
            # Emergency and listening controls
            ("stop listening", "stop_listening", {}),
            ("pause listening", "pause_listening", {}),
            ("resume listening", "resume_listening", {}),
            ("cancel the current command", "cancel_action", {}),
            ("cancel pending actions", "cancel_action", {}),
            ("return to idle mode", "cancel_action", {}),
        ]

        inserted = 0
        for text, intent, entities in baseline_dataset:
            ok, _, _ = self.add_example(text, intent, entities, approved=True, dataset_version=1)
            if ok:
                inserted += 1

        logger.info(f"Seeded {inserted} baseline training examples.")
        return inserted
