"""Model Registry and Version Management for Jarvis.
Handles versioned checkpoints, rollbacks, and active model metadata persistence.
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import torch

try:
    from config import MODELS_DIR
    from storage.database import DatabaseManager
except (ImportError, ValueError):
    from jarvis.config import MODELS_DIR
    from jarvis.storage.database import DatabaseManager
from .model import JarvisIntentNet
from .tokenizer import JarvisTokenizer

logger = logging.getLogger(__name__)


class ModelRegistry:
    """Manages versioned PyTorch model checkpoints, vocabularies, and rollbacks."""

    def __init__(self, models_dir: Optional[Path] = None, db_manager: Optional[DatabaseManager] = None):
        self.models_dir = Path(models_dir) if models_dir else MODELS_DIR
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.db = db_manager or DatabaseManager()

    def get_latest_version(self) -> int:
        """Returns the highest saved version number, minimum 0."""
        conn = self.db.get_connection()
        cursor = conn.execute("SELECT MAX(version) FROM model_checkpoints")
        val = cursor.fetchone()[0]
        return val if val is not None else 0

    def get_active_version(self) -> Optional[int]:
        """Returns the currently active model version number."""
        conn = self.db.get_connection()
        cursor = conn.execute("SELECT version FROM model_checkpoints WHERE is_active = 1 LIMIT 1")
        row = cursor.fetchone()
        if row:
            return row["version"]
        return self.get_latest_version() or None

    def save_checkpoint(
        self,
        model: JarvisIntentNet,
        tokenizer: JarvisTokenizer,
        label_to_idx: Dict[str, int],
        idx_to_label: Dict[int, str],
        dataset_size: int,
        accuracy: float,
        loss: float,
    ) -> Tuple[int, Path, Path]:
        """Saves weights, tokenizer, and metadata for a new model version."""
        new_version = self.get_latest_version() + 1
        version_dir = self.models_dir / f"v{new_version}"
        version_dir.mkdir(parents=True, exist_ok=True)

        model_path = version_dir / f"model_v{new_version}.pt"
        vocab_path = version_dir / f"vocab_v{new_version}.json"
        meta_path = version_dir / f"metadata_v{new_version}.json"

        # 1. Save PyTorch weights and hyperparams
        torch.save(
            {
                "state_dict": model.state_dict(),
                "vocab_size": model.vocab_size,
                "num_classes": model.num_classes,
                "embedding_dim": model.embedding_dim,
                "hidden_dim": model.hidden_dim,
            },
            model_path,
        )

        # 2. Save Tokenizer vocabulary
        tokenizer.save(vocab_path)

        # 3. Save Metadata
        now_iso = datetime.now(timezone.utc).isoformat()
        metadata = {
            "version": new_version,
            "created_at": now_iso,
            "dataset_size": dataset_size,
            "accuracy": accuracy,
            "loss": loss,
            "label_to_idx": label_to_idx,
            "idx_to_label": {str(k): v for k, v in idx_to_label.items()},
        }
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        # 4. Record in database and set active
        conn = self.db.get_connection()
        with conn:
            conn.execute("UPDATE model_checkpoints SET is_active = 0")
            conn.execute(
                """
                INSERT INTO model_checkpoints (
                    version, file_path, vocab_path, dataset_size, accuracy, loss, created_at, is_active
                ) VALUES (?, ?, ?, ?, ?, ?, ?, 1)
                """,
                (
                    new_version,
                    str(model_path),
                    str(vocab_path),
                    dataset_size,
                    accuracy,
                    loss,
                    now_iso,
                ),
            )

        logger.info(f"Saved model checkpoint version {new_version} to {version_dir}")
        return new_version, model_path, vocab_path

    def load_version(
        self, version: int, device: Optional[torch.device] = None
    ) -> Optional[Tuple[JarvisIntentNet, JarvisTokenizer, Dict[int, str], Dict[str, Any]]]:
        """Loads a specific model version, tokenizer, and label mappings."""
        target_device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        version_dir = self.models_dir / f"v{version}"
        model_path = version_dir / f"model_v{version}.pt"
        vocab_path = version_dir / f"vocab_v{version}.json"
        meta_path = version_dir / f"metadata_v{version}.json"

        if not (model_path.exists() and vocab_path.exists() and meta_path.exists()):
            logger.warning(f"Version {version} checkpoint files missing at {version_dir}")
            return None

        try:
            # Metadata
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            idx_to_label = {int(k): v for k, v in meta["idx_to_label"].items()}

            # Tokenizer
            tokenizer = JarvisTokenizer.load(vocab_path)

            # PyTorch Model
            checkpoint = torch.load(model_path, map_location=target_device)
            model = JarvisIntentNet(
                vocab_size=checkpoint["vocab_size"],
                num_classes=checkpoint["num_classes"],
                embedding_dim=checkpoint.get("embedding_dim", 64),
                hidden_dim=checkpoint.get("hidden_dim", 64),
            )
            model.load_state_dict(checkpoint["state_dict"])
            model.to(target_device)
            model.eval()

            return model, tokenizer, idx_to_label, meta
        except Exception as e:
            logger.error(f"Failed to load checkpoint version {version}: {e}")
            return None

    def rollback(self) -> Tuple[bool, str, Optional[int]]:
        """Rolls back the active model to the immediate preceding version."""
        current_active = self.get_active_version()
        if not current_active or current_active <= 1:
            return False, "Cannot rollback: No previous model version exists.", None

        previous_version = current_active - 1
        loaded = self.load_version(previous_version)
        if not loaded:
            return False, f"Previous checkpoint version {previous_version} is missing or corrupted.", None

        conn = self.db.get_connection()
        with conn:
            conn.execute("UPDATE model_checkpoints SET is_active = 0")
            conn.execute("UPDATE model_checkpoints SET is_active = 1 WHERE version = ?", (previous_version,))

        return True, f"Model successfully rolled back to version {previous_version}.", previous_version

    def list_checkpoints(self) -> List[Dict[str, Any]]:
        """Lists all registered checkpoints."""
        conn = self.db.get_connection()
        cursor = conn.execute("SELECT * FROM model_checkpoints ORDER BY version ASC")
        rows = cursor.fetchall()
        return [dict(r) for r in rows]
