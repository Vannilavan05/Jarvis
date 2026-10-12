"""Custom Voice Commands Plugin for Jarvis.
Allows users to register and trigger custom voice macro mappings safely.
"""

import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List

try:
    from security.permissions import PermissionLevel
except (ImportError, ValueError):
    from jarvis.security.permissions import PermissionLevel
from .base import BasePlugin, PluginExecutionResult

logger = logging.getLogger(__name__)


class CustomCommandsPlugin(BasePlugin):
    """Executes and manages user-defined voice macros."""

    @property
    def plugin_id(self) -> str:
        return "custom_commands"

    @property
    def name(self) -> str:
        return "Custom Commands"

    @property
    def version(self) -> str:
        return "1.0.0"

    @property
    def supported_intents(self) -> List[str]:
        return ["custom_command"]

    @property
    def required_permissions(self) -> Dict[str, PermissionLevel]:
        return {"custom_command": PermissionLevel.STANDARD}

    def execute(
        self, intent: str, parameters: Dict[str, Any], context: Dict[str, Any]
    ) -> PluginExecutionResult:
        phrase = parameters.get("phrase", "").strip().lower()
        db = context.get("db_manager")
        if not db:
            return PluginExecutionResult(
                success=False, spoken_response="Database connection unavailable.", error="No db"
            )

        conn = db.get_connection()
        cursor = conn.execute(
            "SELECT intent, parameters FROM custom_commands WHERE phrase = ? COLLATE NOCASE", (phrase,)
        )
        row = cursor.fetchone()
        if not row:
            return PluginExecutionResult(
                success=False,
                spoken_response=f"No custom command found matching '{phrase}'.",
                error="Custom command not found",
            )

        target_intent = row["intent"]
        target_params = json.loads(row["parameters"] or "{}")

        return PluginExecutionResult(
            success=True,
            spoken_response=f"Executing custom macro for {phrase}.",
            data={"target_intent": target_intent, "target_params": target_params},
        )
