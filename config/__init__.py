"""Jarvis Configuration Module.
Provides centralized path constants and configuration loading.
"""

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
CONFIG_DIR = BASE_DIR / "config"
MODELS_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data"
LOGS_DIR = BASE_DIR / "logs"

SETTINGS_PATH = CONFIG_DIR / "settings.json"
DATABASE_PATH = DATA_DIR / "jarvis.db"
