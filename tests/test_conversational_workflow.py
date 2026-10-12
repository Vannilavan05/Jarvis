"""End-to-End Tests for Conversational Voice Workflows, Training Modes, and Security Defenses.
"""

import tempfile
from pathlib import Path
import pytest

from ai.model_registry import ModelRegistry
from ai.predictor import IntentPredictor
from ai.trainer import ModelTrainer
from execution.command_registry import CommandRegistry
from execution.dispatcher import CommandDispatcher
from execution.plugin_manager import PluginManager
from nlp.intent_parser import IntentParser
from security.audit_logger import AuditLogger
from security.confirmation_manager import ConfirmationManager
from security.permissions import PermissionLevel
from storage.database import DatabaseManager
from storage.settings_store import SettingsStore
from storage.training_store import TrainingStore


@pytest.fixture
def test_system():
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        db_path = root / "test.db"
        cfg_path = root / "settings.json"
        models_dir = root / "models"

        db = DatabaseManager(db_path)
        settings = SettingsStore(cfg_path)
        training_store = TrainingStore(db)
        training_store.seed_default_examples()

        registry = ModelRegistry(models_dir=models_dir, db_manager=db)
        # Train baseline
        trainer = ModelTrainer(registry=registry)
        trainer.train(training_store.get_all_examples(), epochs=4)

        predictor = IntentPredictor(registry=registry, confidence_threshold=0.5)
        parser = IntentParser(predictor=predictor, confidence_threshold=0.5)
        confirm_mgr = ConfirmationManager(default_timeout_seconds=10.0)
        audit = AuditLogger(db)
        plugins = PluginManager()

        context = {
            "settings": settings,
            "training_store": training_store,
            "model_registry": registry,
            "predictor": predictor,
            "db_manager": db,
        }

        dispatcher = CommandDispatcher(
            command_registry=CommandRegistry(),
            plugin_manager=plugins,
            confirmation_manager=confirm_mgr,
            audit_logger=audit,
            current_permission_level=PermissionLevel.PRIVILEGED,
            context=context,
        )

        yield {
            "parser": parser,
            "dispatcher": dispatcher,
            "training_store": training_store,
            "registry": registry,
            "confirm_mgr": confirm_mgr,
            "db": db,
        }
        db.close()


def test_spoken_training_workflow_end_to_end(test_system):
    """Tests: User speaks:
    1. 'Jarvis, enter training mode'
    2. 'When I say launch browser, open Chrome'
    3. 'Jarvis, train the model'
    4. 'Jarvis, evaluate the model'
    """
    parser = test_system["parser"]
    dispatcher = test_system["dispatcher"]
    training_store = test_system["training_store"]
    registry = test_system["registry"]

    # Step 1: Enter training mode
    cmd1 = parser.parse("Jarvis, enter training mode")
    assert cmd1.intent == "start_training"
    resp1 = dispatcher.dispatch(cmd1)
    assert resp1.success is True
    assert "training mode activated" in resp1.spoken_feedback.lower()

    # Step 2: Add training example via speech syntax
    cmd2 = parser.parse("when I say launch my browser, open chrome")
    assert cmd2.intent == "add_example"
    resp2 = dispatcher.dispatch(cmd2)
    assert resp2.success is True
    assert "saved" in resp2.spoken_feedback.lower()

    # Verify example in database
    last_ex = training_store.get_last_example()
    assert last_ex is not None
    assert "launch my browser" in last_ex["text"].lower()

    # Step 3: Train model
    initial_ver = registry.get_active_version()
    cmd3 = parser.parse("Jarvis, train the model")
    assert cmd3.intent == "train_model"
    resp3 = dispatcher.dispatch(cmd3)
    assert resp3.success is True
    assert "training complete" in resp3.spoken_feedback.lower()
    assert registry.get_active_version() > initial_ver

    # Step 4: Evaluate model
    cmd4 = parser.parse("Jarvis, evaluate the model")
    assert cmd4.intent == "evaluate_model"
    resp4 = dispatcher.dispatch(cmd4)
    assert resp4.success is True
    assert "accuracy" in resp4.spoken_feedback.lower()


def test_prevention_of_external_content_code_injection(test_system):
    """Verifies that malicious payloads inside command arguments cannot execute code or shell commands."""
    parser = test_system["parser"]
    dispatcher = test_system["dispatcher"]

    # User searches for shell injection string: should just format safe URL, never execute shell
    cmd = parser.parse("Jarvis search the web for rm -rf / ; shutdown /s")
    assert cmd.intent == "web_search"
    resp = dispatcher.dispatch(cmd)
    assert resp.success is True
    # Verify that it remained purely data in a google search query URL
    assert "google.com" in resp.data["url"]
    assert "shutdown" in resp.data["url"]


def test_no_startup_automatic_command_execution(test_system):
    """Verifies that audit logs and database existence do not trigger autonomous execution on startup."""
    dispatcher = test_system["dispatcher"]
    # At startup, no actions should be pending in confirmation manager
    assert dispatcher.confirmation_mgr.has_pending() is False
