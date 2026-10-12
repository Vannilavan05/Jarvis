"""System Controls Plugin for Jarvis.
Manages audio volume adjustments, screen lock, and system power states.
"""

import ctypes
import logging
import subprocess
from typing import Any, Dict, List

try:
    from security.permissions import PermissionLevel
except (ImportError, ValueError):
    from jarvis.security.permissions import PermissionLevel
from .base import BasePlugin, PluginExecutionResult

logger = logging.getLogger(__name__)


class SystemControlsPlugin(BasePlugin):
    """Controls volume, locks desktop, and handles power actions."""

    # Windows Virtual Key Codes for multimedia
    VK_VOLUME_MUTE = 0xAD
    VK_VOLUME_DOWN = 0xAE
    VK_VOLUME_UP = 0xAF
    KEYEVENTF_EXTENDEDKEY = 0x0001
    KEYEVENTF_KEYUP = 0x0002

    @property
    def plugin_id(self) -> str:
        return "system_controls"

    @property
    def name(self) -> str:
        return "System Controls"

    @property
    def version(self) -> str:
        return "1.0.0"

    @property
    def supported_intents(self) -> List[str]:
        return ["system_control", "reboot_system", "shutdown_system"]

    @property
    def required_permissions(self) -> Dict[str, PermissionLevel]:
        return {
            "system_control": PermissionLevel.STANDARD,
            "reboot_system": PermissionLevel.PRIVILEGED,
            "shutdown_system": PermissionLevel.PRIVILEGED,
        }

    def _press_key(self, vk_code: int):
        """Simulates keypress for Windows multimedia keys."""
        try:
            ctypes.windll.user32.keybd_event(vk_code, 0, self.KEYEVENTF_EXTENDEDKEY, 0)
            ctypes.windll.user32.keybd_event(vk_code, 0, self.KEYEVENTF_EXTENDEDKEY | self.KEYEVENTF_KEYUP, 0)
        except Exception as e:
            logger.warning(f"Could not dispatch key event: {e}")

    def execute(
        self, intent: str, parameters: Dict[str, Any], context: Dict[str, Any]
    ) -> PluginExecutionResult:
        if intent == "system_control":
            action = parameters.get("action", "unknown").lower()

            if action == "volume_up":
                for _ in range(5):
                    self._press_key(self.VK_VOLUME_UP)
                return PluginExecutionResult(success=True, spoken_response="Increasing volume.")

            elif action == "volume_down":
                for _ in range(5):
                    self._press_key(self.VK_VOLUME_DOWN)
                return PluginExecutionResult(success=True, spoken_response="Decreasing volume.")

            elif action in {"mute", "unmute"}:
                self._press_key(self.VK_VOLUME_MUTE)
                state = "Muted" if action == "mute" else "Toggled mute"
                return PluginExecutionResult(success=True, spoken_response=f"{state} audio.")

            elif action == "lock":
                try:
                    ctypes.windll.user32.LockWorkStation()
                    return PluginExecutionResult(success=True, spoken_response="Locking your workstation.")
                except Exception as e:
                    return PluginExecutionResult(
                        success=False, spoken_response="Failed to lock workstation.", error=str(e)
                    )

            return PluginExecutionResult(
                success=False,
                spoken_response=f"Unsupported system control action '{action}'.",
                error="Unknown action",
            )

        elif intent == "shutdown_system":
            try:
                subprocess.Popen(["shutdown", "/s", "/t", "60"], shell=False)
                return PluginExecutionResult(
                    success=True,
                    spoken_response="System shutdown scheduled in 60 seconds. Say cancel to abort.",
                )
            except Exception as e:
                return PluginExecutionResult(
                    success=False, spoken_response="Failed to initiate shutdown.", error=str(e)
                )

        elif intent == "reboot_system":
            try:
                subprocess.Popen(["shutdown", "/r", "/t", "60"], shell=False)
                return PluginExecutionResult(
                    success=True,
                    spoken_response="System reboot scheduled in 60 seconds. Say cancel to abort.",
                )
            except Exception as e:
                return PluginExecutionResult(
                    success=False, spoken_response="Failed to initiate reboot.", error=str(e)
                )

        return PluginExecutionResult(success=False, spoken_response="Unsupported system control operation.")
