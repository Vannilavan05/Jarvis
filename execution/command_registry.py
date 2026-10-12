"""Command Registry for Jarvis.
Maintains the master list of approved intents, parameters, and permission requirements.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional

try:
    from security.permissions import INTENT_PERMISSION_MAP, PermissionLevel
except (ImportError, ValueError):
    from jarvis.security.permissions import INTENT_PERMISSION_MAP, PermissionLevel


@dataclass
class IntentMetadata:
    intent: str
    description: str
    required_parameters: List[str]
    permission_level: PermissionLevel
    requires_confirmation: bool


class CommandRegistry:
    """Master registry of all valid and approved intents in Jarvis."""

    def __init__(self):
        self._intents: Dict[str, IntentMetadata] = {}
        self._register_defaults()

    def register(self, intent: str, description: str, required_parameters: Optional[List[str]] = None,
                 permission_level: Optional[PermissionLevel] = None, requires_confirmation: bool = False):
        """Registers an intent definition."""
        perm = permission_level or INTENT_PERMISSION_MAP.get(intent, PermissionLevel.STANDARD)
        self._intents[intent] = IntentMetadata(intent, description, required_parameters or [], perm, requires_confirmation)

    def get(self, intent: str) -> Optional[IntentMetadata]:
        return self._intents.get(intent)

    def is_valid_intent(self, intent: str) -> bool:
        return intent in self._intents

    def all_intents(self) -> Dict[str, IntentMetadata]:
        return self._intents.copy()

    def _register_defaults(self):
        self.register("open_application", "Opens an installed application", ["application"], PermissionLevel.STANDARD)
        self.register("close_application", "Terminates an application process", ["application"], PermissionLevel.DESTRUCTIVE, True)
        self.register("get_active_window", "Identifies the currently focused application window", [], PermissionLevel.READ_ONLY)
        self.register("click_ui_control", "Clicks a uniquely named accessible control", ["control_name"], PermissionLevel.STANDARD, True)
        self.register("type_text", "Types user-provided text into the verified focused control", ["text"], PermissionLevel.STANDARD)
        self.register("press_hotkey", "Presses a permitted keyboard shortcut", ["keys"], PermissionLevel.STANDARD)
        self.register("web_search", "Performs a web search via the default browser", ["query"], PermissionLevel.STANDARD)
        self.register("read_file", "Reads a file from authorized directories", ["filename"], PermissionLevel.READ_ONLY)
        self.register("create_file", "Creates a file in authorized directories", ["filename"], PermissionLevel.FILE_MODIFICATION)
        self.register("delete_file", "Deletes a file in authorized directories", ["filename"], PermissionLevel.DESTRUCTIVE, True)
        self.register("list_files", "Lists files in authorized directories", [], PermissionLevel.READ_ONLY)
        self.register("system_info", "Retrieves system resource metrics", [], PermissionLevel.READ_ONLY)
        self.register("system_control", "Adjusts volume or locks workstation", [], PermissionLevel.STANDARD)
        self.register("reboot_system", "Reboots operating system", [], PermissionLevel.PRIVILEGED, True)
        self.register("shutdown_system", "Shuts down operating system", [], PermissionLevel.PRIVILEGED, True)
        self.register("start_training", "Enters conversational training mode", [], PermissionLevel.STANDARD)
        self.register("add_example", "Adds a training example to local dataset", ["phrase"], PermissionLevel.STANDARD)
        self.register("review_examples", "Reviews existing training dataset", [], PermissionLevel.READ_ONLY)
        self.register("delete_last_example", "Removes most recent training example", [], PermissionLevel.DESTRUCTIVE, True)
        self.register("train_model", "Trains the PyTorch neural intent model", [], PermissionLevel.STANDARD)
        self.register("evaluate_model", "Evaluates model accuracy on test split", [], PermissionLevel.READ_ONLY)
        self.register("get_model_version", "Reports active model version", [], PermissionLevel.READ_ONLY)
        self.register("rollback_model", "Rolls back to previous model checkpoint", [], PermissionLevel.DESTRUCTIVE, True)
        self.register("stop_listening", "Halts voice listening", [], PermissionLevel.READ_ONLY)
        self.register("pause_listening", "Pauses voice listening", [], PermissionLevel.READ_ONLY)
        self.register("resume_listening", "Resumes voice listening", [], PermissionLevel.READ_ONLY)
        self.register("cancel_action", "Cancels pending command or actions", [], PermissionLevel.READ_ONLY)
        self.register("confirm_action", "Confirms a pending sensitive action", [], PermissionLevel.STANDARD)
        self.register("custom_command", "Executes registered custom macro", ["phrase"], PermissionLevel.STANDARD)
        # Explicit local memory and automation commands.
        self.register("remember_fact", "Stores an explicit user-approved memory locally", ["memory_value"], PermissionLevel.STANDARD)
        self.register("recall_memories", "Reads saved local user memories", [], PermissionLevel.READ_ONLY)
        self.register("get_task_status", "Reports saved automation task progress", [], PermissionLevel.READ_ONLY)
        self.register("get_recent_history", "Reads recent saved conversation history", [], PermissionLevel.READ_ONLY)
        self.register("run_automation", "Executes a validated sequence of approved actions", ["steps"], PermissionLevel.STANDARD)
