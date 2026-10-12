"""Tests for Security Validator, Confirmation Manager, Path Traversal Defense, and Audit Logging.
"""

import tempfile
import time
from pathlib import Path
import pytest

from security.audit_logger import AuditLogger
from security.confirmation_manager import ConfirmationManager
from security.validator import SecurityValidator
from storage.database import DatabaseManager


def test_path_traversal_defense():
    with tempfile.TemporaryDirectory() as tmpdir:
        allowed_dir = Path(tmpdir) / "safe_zone"
        allowed_dir.mkdir()

        # Legitimate file inside safe zone
        safe_file = allowed_dir / "notes.txt"
        is_safe, _ = SecurityValidator.is_safe_path(safe_file, [allowed_dir])
        assert is_safe is True

        # Path traversal attack attempting to escape
        traversal_file = allowed_dir / ".." / "unauthorized.txt"
        is_safe, msg = SecurityValidator.is_safe_path(traversal_file, [allowed_dir])
        assert is_safe is False
        assert "Access denied" in msg


def test_filename_sanitization():
    dirty = "my..evil/file:name*?.txt"
    cleaned = SecurityValidator.sanitize_filename(dirty)
    assert "/" not in cleaned
    assert "\\" not in cleaned
    assert ":" not in cleaned
    assert "*" not in cleaned


def test_command_argument_injection_defense():
    assert SecurityValidator.is_safe_argument("notepad") is True
    assert SecurityValidator.is_safe_argument("calc; rm -rf /") is False
    assert SecurityValidator.is_safe_argument("test | dir") is False
    assert SecurityValidator.is_safe_argument("arg & evil") is False


def test_confirmation_manager_lifecycle_and_invalidation():
    cm = ConfirmationManager(default_timeout_seconds=5.0)

    executed = {"val": False}

    def sample_callback():
        executed["val"] = True
        return "Done"

    # Request confirmation
    action_id, prompt = cm.request_confirmation(
        intent="delete_file",
        target_description="delete temporary.txt",
        callback=sample_callback,
        parameters={"filename": "temporary.txt"},
    )
    assert cm.has_pending() is True
    assert "delete temporary.txt" in prompt

    # Confirm action
    ok, msg, res = cm.verify_and_execute("confirm deletion")
    assert ok is True
    assert executed["val"] is True
    assert res == "Done"

    # Token must be invalidated immediately (no replay attack)
    assert cm.has_pending() is False
    ok2, _, _ = cm.verify_and_execute("confirm deletion")
    assert ok2 is False


def test_confirmation_timeout_expiration():
    cm = ConfirmationManager(default_timeout_seconds=0.1)

    cm.request_confirmation(
        intent="delete_file",
        target_description="delete expired.txt",
        callback=lambda: None,
        parameters={},
        timeout_seconds=0.05,
    )

    time.sleep(0.08)
    # Should be expired
    assert cm.has_pending() is False
    ok, msg, _ = cm.verify_and_execute("confirm")
    assert ok is False
    assert "no pending actions" in msg.lower()


def test_audit_logging():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.db"
        db = DatabaseManager(db_path)
        logger = AuditLogger(db)

        log_id = logger.log(
            command_text="open chrome",
            intent="open_application",
            confidence=0.98,
            permission_level="STANDARD",
            authorized=True,
            status="COMPLETED",
            result="Opening chrome.",
            model_version=1,
        )
        assert log_id > 0

        recent = logger.get_recent(limit=5)
        assert len(recent) == 1
        assert recent[0]["command_text"] == "open chrome"
        assert recent[0]["authorized"] is True

        # Clear
        cleared = logger.clear()
        assert cleared == 1
        assert len(logger.get_recent()) == 0
        db.close()
