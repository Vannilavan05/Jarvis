"""Tests for Model Training, Evaluation, Checkpoints, and Rollback.
"""

import tempfile
from pathlib import Path
import pytest

from ai.dataset import DatasetManager
from ai.model_registry import ModelRegistry
from ai.tokenizer import JarvisTokenizer
from ai.trainer import ModelTrainer
from storage.database import DatabaseManager


def test_dataset_splitting_and_validation():
    tokenizer = JarvisTokenizer()

    # Small dataset (< 4) should be rejected
    too_few = [{"text": "hello", "intent": "greet", "approved": True}]
    splits = DatasetManager.prepare_datasets(too_few, tokenizer)
    assert splits.is_sufficient is False

    # Sufficient dataset
    examples = [
        {"text": f"command {i}", "intent": f"intent_{i % 3}", "approved": True}
        for i in range(12)
    ]
    splits = DatasetManager.prepare_datasets(examples, tokenizer)
    assert splits.is_sufficient is True
    assert len(splits.train_dataset) > 0
    assert splits.val_dataset is not None


def test_model_training_and_rollback():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.db"
        models_dir = Path(tmpdir) / "models"
        db = DatabaseManager(db_path)
        registry = ModelRegistry(models_dir=models_dir, db_manager=db)

        # Create training examples across 3 intents
        examples = [
            {"text": "open chrome browser", "intent": "open_application", "approved": True},
            {"text": "launch google chrome", "intent": "open_application", "approved": True},
            {"text": "search the web for news", "intent": "web_search", "approved": True},
            {"text": "google latest developments", "intent": "web_search", "approved": True},
            {"text": "read file summary.txt", "intent": "read_file", "approved": True},
            {"text": "show document contents", "intent": "read_file", "approved": True},
            {"text": "check cpu status", "intent": "system_info", "approved": True},
            {"text": "what is memory usage", "intent": "system_info", "approved": True},
        ]

        trainer = ModelTrainer(registry=registry)

        # Train version 1
        res1 = trainer.train(examples, epochs=5)
        assert res1.success is True
        assert res1.version == 1
        assert registry.get_active_version() == 1

        # Train version 2
        res2 = trainer.train(examples, epochs=5)
        assert res2.success is True
        assert res2.version == 2
        assert registry.get_active_version() == 2

        # Test Rollback from version 2 to version 1
        ok, msg, prev_v = registry.rollback()
        assert ok is True
        assert prev_v == 1
        assert registry.get_active_version() == 1
        db.close()
