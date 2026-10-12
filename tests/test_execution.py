"""Tests for Command Dispatcher Lifecycle and Confirmation Flow.
"""

from unittest.mock import MagicMock
import pytest

from execution.dispatcher import CommandDispatcher
from nlp.intent_parser import CommandObject
from security.confirmation_manager import ConfirmationManager
from security.permissions import PermissionLevel


def test_dispatcher_rejects_ambiguous_command():
    dispatcher = CommandDispatcher()
    cmd = CommandObject(
        intent="unknown",
        parameters={},
        confidence=0.2,
        is_ambiguous=True,
        clarification_prompt="Could you clarify your request?",
        raw_text="blah blah blah",
    )
    resp = dispatcher.dispatch(cmd)
    assert resp.success is False
    assert "clarify" in resp.spoken_feedback.lower()


def test_dispatcher_rejects_unregistered_intent():
    dispatcher = CommandDispatcher()
    cmd = CommandObject(
        intent="unregistered_cyber_strike",
        parameters={},
        confidence=0.99,
        is_ambiguous=False,
        raw_text="cyber strike",
    )
    resp = dispatcher.dispatch(cmd)
    assert resp.success is False
    assert "not registered" in resp.spoken_feedback.lower()


def test_dispatcher_sensitive_action_requires_confirmation():
    cm = ConfirmationManager()
    dispatcher = CommandDispatcher(confirmation_manager=cm)

    # delete_file is DESTRUCTIVE, should require confirmation
    cmd = CommandObject(
        intent="delete_file",
        parameters={"filename": "test.txt"},
        confidence=0.95,
        raw_text="delete file test.txt",
    )

    resp = dispatcher.dispatch(cmd)
    assert resp.success is True
    assert resp.requires_confirmation is True
    assert cm.has_pending() is True
    assert "confirm" in resp.spoken_feedback.lower()


def test_dispatcher_emergency_stop():
    dispatcher = CommandDispatcher()
    cmd = CommandObject(
        intent="stop_listening",
        parameters={},
        confidence=1.0,
        is_emergency=True,
        raw_text="stop listening",
    )
    resp = dispatcher.dispatch(cmd)
    assert resp.success is True
    assert resp.is_emergency is True
    assert "paused" in resp.spoken_feedback.lower() or "deactivating" in resp.spoken_feedback.lower()
