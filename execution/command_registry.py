"""Command Registry for Jarvis.
Maintains the master list of approved intents, metadata, parameters, and permission requirements.
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

    def register(
        self,
        intent: str,
        description: str,
        required_parameters: Optional[List[str]] = None,
        permission_level: Optional[PermissionLevel] = None,
        requires_confirmation: bool = False,
    ):
        """Registers an intent definition."""
        perm = permission_level or INTENT_PERMISSION_MAP.get(intent, PermissionLevel.STANDARD)
        self._intents[intent] = IntentMetadata(
            intent=intent,
            description=description,
            required_parameters=required_parameters or [],
            permission_level=perm,
            requires_confirmation=requires_confirmation,
        )

    def get(self, intent: str) -> Optional[IntentMetadata]:
        return self._intents.get(intent)

    def is_valid_intent(self, intent: str) -> bool:
        return intent in self._intents

    def all_intents(self) -> Dict[str, IntentMetadata]:
        return self._intents.copy()

    def _register_defaults(self):
        # Applications
        self.register("open_application", "Opens an approved application", ["application"], PermissionLevel.STANDARD)
        self.register("close_application", "Terminates an application process", ["application"], PermissionLevel.DESTRUCTIVE, True)

        # App-agnostic Windows desktop UI Automation
        self.register("get_active_window", "Identifies the currently focused application window", [], PermissionLevel.READ_ONLY)
        self.register("click_ui_control", "Clicks a uniquely named accessible control in the active window", ["control_name"], PermissionLevel.STANDARD, True)
        self.register("type_text", "Types user-provided text into the verified focused control", ["text"], PermissionLevel.STANDARD)
        self.register("press_hotkey", "Presses a permitted keyboard shortcut in the active window", ["keys"], PermissionLevel.STANDARD)

        # Web
        self.register("web_search", "Performs a web search via default browser", ["query"], PermissionLevel.STANDARD)

        # Files
        self.register("read_file", "Reads a file from authorized directories", ["filename"], PermissionLevel.READ_ONLY)
        self.register("create_file", "Creates a file in authorized directories", ["filename"], PermissionLevel.FILE_MODIFICATION)
        self.register("delete_file", "Deletes a file in authorized directories", ["filename"], PermissionLevel.DESTRUCTIVE, True)
        self.register("list_files", "Lists files in authorized directories", [], PermissionLevel.READ_ONLY)

        # System info
        self.register("system_info", "Retrieves system resource metrics", [], PermissionLevel.READ_ONLY)

        # System controls
        self.register("system_control", "Adjusts volume or locks workstation", [], PermissionLevel.STANDARD)
        self.register("reboot_system", "Reboots operating system", [], PermissionLevel.PRIVILEGED, True)
        self.register("shutdown_system", "Shuts down operating system", [], PermissionLevel.PRIVILEGED, True)

        # Model & Training
        self.register("start_training", "Enters conversational training mode", [], PermissionLevel.STANDARD)
        self.register("add_example", "Adds a training example to local dataset", ["phrase"], PermissionLevel.STANDARD)
        self.register("review_examples", "Reviews existing training dataset", [], PermissionLevel.READ_ONLY)
        self.register("delete_last_example", "Removes most recent training example", [], PermissionLevel.DESTRUCTIVE, True)
        self.register("train_model", "Trains the PyTorch neural intent model", [], PermissionLevel.STANDARD)
        self.register("evaluate_model", "Evaluates model accuracy on test split", [], PermissionLevel.READ_ONLY)
        self.register("get_model_version", "Reports active model version", [], PermissionLevel.READ_ONLY)
        self.register("rollback_model", "Rolls back to previous model checkpoint", [], PermissionLevel.DESTRUCTIVE, True)

        # Emergency & Core control
        self.register("stop_listening", "Halts voice listening", [], PermissionLevel.READ_ONLY)
        self.register("pause_listening", "Pauses voice listening", [], PermissionLevel.READ_ONLY)
        self.register("resume_listening", "Resumes voice listening", [], PermissionLevel.READ_ONLY)
        self.register("cancel_action", "Cancels pending command or actions", [], PermissionLevel.READ_ONLY)
        self.register("confirm_action", "Confirms a pending sensitive action", [], PermissionLevel.STANDARD)
        self.register("custom_command", "Executes registered custom macro", ["phrase"], PermissionLevel.STANDARD)
