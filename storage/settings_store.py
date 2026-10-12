"""Settings Store for Jarvis.
Manages application configuration, persistence, and runtime updates.
"""

import json
import logging
import os
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from config import SETTINGS_PATH
except (ImportError, ValueError):
    from jarvis.config import SETTINGS_PATH

logger = logging.getLogger(__name__)


class SettingsStore:
    """Manages reading and writing application settings to JSON."""

    def __init__(self, config_path: Optional[Path] = None):
        self.config_path = Path(config_path) if config_path else SETTINGS_PATH
        self._lock = threading.Lock()
        self._settings: Dict[str, Any] = {}
        self.load()

    def load(self) -> Dict[str, Any]:
        """Loads settings from JSON file or generates defaults if absent."""
        with self._lock:
            if not self.config_path.exists():
                self._settings = self._get_default_settings()
                self._save_internal()
            else:
                try:
                    with open(self.config_path, "r", encoding="utf-8") as f:
                        self._settings = json.load(f)
                except Exception as e:
                    logger.error(f"Failed to load settings from {self.config_path}: {e}")
                    self._settings = self._get_default_settings()
            return self._settings.copy()

    def get(self, key: str, default: Any = None) -> Any:
        """Retrieves a setting by key."""
        with self._lock:
            return self._settings.get(key, default)

    def set(self, key: str, value: Any, persist: bool = True) -> None:
        """Updates a setting and optionally persists it."""
        with self._lock:
            self._settings[key] = value
            if persist:
                self._save_internal()

    def update(self, updates: Dict[str, Any], persist: bool = True) -> None:
        """Updates multiple settings at once."""
        with self._lock:
            self._settings.update(updates)
            if persist:
                self._save_internal()

    def all(self) -> Dict[str, Any]:
        """Returns a copy of all current settings."""
        with self._lock:
            return self._settings.copy()

    def get_allowed_directories(self) -> List[Path]:
        """Returns canonical Path objects of allowed file directories."""
        raw_dirs = self.get("allowed_file_directories", [])
        resolved = []
        project_root = self.config_path.resolve().parent.parent
        for directory in raw_dirs:
            try:
                # Support user-home paths and project-relative paths on any machine.
                path = Path(str(directory)).expanduser()
                if not path.is_absolute():
                    path = project_root / path
                path = path.resolve()
                if path not in resolved:
                    resolved.append(path)
            except (OSError, RuntimeError, TypeError, ValueError):
                logger.warning("Ignoring invalid allowed directory setting: %r", directory)
        return resolved

    def _save_internal(self) -> None:
        """Internal save method without lock re-acquisition."""
        try:
            self.config_path.parent.mkdir(parents=True, exist_ok=True)
            temp_path = self.config_path.with_suffix(".tmp")
            with open(temp_path, "w", encoding="utf-8") as f:
                json.dump(self._settings, f, indent=2)
            temp_path.replace(self.config_path)
        except Exception as e:
            logger.error(f"Failed to persist settings to {self.config_path}: {e}")

    def _get_default_settings(self) -> Dict[str, Any]:
        return {
            "assistant_name": "Jarvis",
            "wake_word": "jarvis",
            "wake_word_aliases": ["jarvis", "hey jarvis", "okay jarvis"],
            "wake_word_sensitivity": 0.7,
            "sample_rate": 16000,
            "audio_device_index": None,
            "vosk_model_path": "models/vosk-model-small-en-us-0.15",
            "nlp_confidence_threshold": 0.65,
            "confirmation_timeout_seconds": 30,
            "allowed_file_directories": [
                str(Path.home() / "Documents"),
                str(Path.home() / "Downloads"),
                str(Path.home() / "Desktop"),
            ],
            "tts_rate": 180,
            "tts_volume": 1.0,
            "tts_voice_id": None,
            "ui_theme": "holographic_cyan",
            "require_confirmation_for_destructive": True,
            "active_model_version": 1,
            "dataset_version": 1,
            "plugins_enabled": [
                "applications",
                "files",
                "web_search",
                "system_info",
                "system_controls",
                "model_management",
                "custom_commands",
            ],
            "approved_applications": {
                "chrome": "chrome",
                "browser": "chrome",
                "notepad": "notepad",
                "calculator": "calc",
                "explorer": "explorer",
                "code": "code",
                "cmd": "cmd",
                "terminal": "wt",
            },
        }
