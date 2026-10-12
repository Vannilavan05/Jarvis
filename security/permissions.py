"""Permission Definitions and Intent Security Mapping for Jarvis.
"""

from enum import Enum
from typing import Dict


class PermissionLevel(str, Enum):
    """Hierarchy of security permissions."""
    READ_ONLY = "READ_ONLY"
    STANDARD = "STANDARD"
    FILE_MODIFICATION = "FILE_MODIFICATION"
    DESTRUCTIVE = "DESTRUCTIVE"
    PRIVILEGED = "PRIVILEGED"

    @property
    def rank(self) -> int:
        ranks = {
            "READ_ONLY": 1,
            "STANDARD": 2,
            "FILE_MODIFICATION": 3,
            "DESTRUCTIVE": 4,
            "PRIVILEGED": 5,
        }
        return ranks.get(self.value, 99)

    def is_authorized(self, required: "PermissionLevel") -> bool:
        """Returns True if self meets or exceeds the required permission level."""
        return self.rank >= required.rank


# Default mapping from intent to required security level
INTENT_PERMISSION_MAP: Dict[str, PermissionLevel] = {
    # Read-only
    "system_info": PermissionLevel.READ_ONLY,
    "read_file": PermissionLevel.READ_ONLY,
    "list_files": PermissionLevel.READ_ONLY,
    "review_examples": PermissionLevel.READ_ONLY,
    "get_model_version": PermissionLevel.READ_ONLY,
    "evaluate_model": PermissionLevel.READ_ONLY,
    "pause_listening": PermissionLevel.READ_ONLY,
    "resume_listening": PermissionLevel.READ_ONLY,
    "stop_listening": PermissionLevel.READ_ONLY,
    "cancel_action": PermissionLevel.READ_ONLY,

    # Standard
    "open_application": PermissionLevel.STANDARD,
    "web_search": PermissionLevel.STANDARD,
    "system_control": PermissionLevel.STANDARD,
    "start_training": PermissionLevel.STANDARD,
    "add_example": PermissionLevel.STANDARD,
    "train_model": PermissionLevel.STANDARD,
    "custom_command": PermissionLevel.STANDARD,

    # File modification
    "create_file": PermissionLevel.FILE_MODIFICATION,

    # Destructive (requires explicit spoken confirmation)
    "delete_file": PermissionLevel.DESTRUCTIVE,
    "rollback_model": PermissionLevel.DESTRUCTIVE,
    "close_application": PermissionLevel.DESTRUCTIVE,

    # Privileged
    "reboot_system": PermissionLevel.PRIVILEGED,
    "shutdown_system": PermissionLevel.PRIVILEGED,
}
