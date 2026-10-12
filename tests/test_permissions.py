"""Tests for Permission Hierarchy and Intent Authorization Enforcement.
"""

import pytest

from execution.command_registry import CommandRegistry
from execution.dispatcher import CommandDispatcher
from nlp.intent_parser import CommandObject
from security.permissions import INTENT_PERMISSION_MAP, PermissionLevel


def test_permission_level_hierarchy():
    read = PermissionLevel.READ_ONLY
    std = PermissionLevel.STANDARD
    destr = PermissionLevel.DESTRUCTIVE
    priv = PermissionLevel.PRIVILEGED

    assert priv.is_authorized(read) is True
    assert priv.is_authorized(destr) is True
    assert std.is_authorized(read) is True
    assert std.is_authorized(destr) is False
    assert read.is_authorized(std) is False


def test_dispatcher_rejects_unauthorized_permission():
    """Verifies that an action requiring higher permissions than current is rejected."""
    # Restrict current dispatcher level to READ_ONLY
    dispatcher = CommandDispatcher(
        current_permission_level=PermissionLevel.READ_ONLY,
    )

    # Attempt to execute standard intent 'open_application'
    cmd = CommandObject(
        intent="open_application",
        parameters={"application": "chrome"},
        confidence=0.95,
        raw_text="open chrome",
    )

    resp = dispatcher.dispatch(cmd)
    assert resp.success is False
    assert "Permission denied" in resp.spoken_feedback
