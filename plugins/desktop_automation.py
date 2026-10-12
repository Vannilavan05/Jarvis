"""System-wide desktop automation for Windows applications.

Uses Windows UI Automation through pywinauto. This is intentionally app-agnostic:
controls are located by accessible UI names rather than per-app plugins.
"""
import logging
import platform
import re
from typing import Any, Dict, List

try:
    from security.permissions import PermissionLevel
except (ImportError, ValueError):
    from jarvis.security.permissions import PermissionLevel
from .base import BasePlugin, PluginExecutionResult

logger = logging.getLogger(__name__)

# Limit keyboard shortcuts to common navigation/editing operations. No shell commands.
SAFE_HOTKEYS = {
    "ctrl+a": "^a", "ctrl+c": "^c", "ctrl+v": "^v", "ctrl+x": "^x",
    "ctrl+z": "^z", "ctrl+y": "^y", "ctrl+s": "^s", "ctrl+f": "^f",
    "ctrl+p": "^p", "ctrl+w": "^w", "ctrl+tab": "^{TAB}",
    "ctrl+shift+tab": "^+{TAB}", "alt+tab": "%{TAB}",
    "enter": "{ENTER}", "escape": "{ESC}", "tab": "{TAB}",
    "shift+tab": "+{TAB}", "home": "{HOME}", "end": "{END}",
    "page up": "{PGUP}", "page down": "{PGDN}",
}

class DesktopAutomationPlugin(BasePlugin):
    """Control the active desktop window without a hard-coded app integration."""

    @property
    def plugin_id(self) -> str:
        return "desktop_automation"

    @property
    def name(self) -> str:
        return "System-wide Desktop Automation"

    @property
    def version(self) -> str:
        return "1.0.0"

    @property
    def supported_intents(self) -> List[str]:
        return ["get_active_window", "click_ui_control", "type_text", "press_hotkey"]

    @property
    def required_permissions(self) -> Dict[str, PermissionLevel]:
        return {
            "get_active_window": PermissionLevel.READ_ONLY,
            "click_ui_control": PermissionLevel.STANDARD,
            "type_text": PermissionLevel.STANDARD,
            "press_hotkey": PermissionLevel.STANDARD,
        }

    def capture_target(self, intent: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Capture the target window before confirmation so focus changes cannot redirect a click."""
        if intent != "click_ui_control":
            return dict(parameters)
        desktop = self._desktop()
        active = desktop.get_active()
        if active is None or not active.exists():
            raise RuntimeError("No active application window could be identified.")
        title = (active.window_text() or "").strip()
        process_id = active.process_id()
        if not title or not isinstance(process_id, int) or process_id <= 0:
            raise RuntimeError("The active window cannot be identified safely.")
        captured = dict(parameters)
        captured["expected_window_title"] = title
        captured["expected_process_id"] = process_id
        return captured

    def _desktop(self):
        if platform.system() != "Windows":
            raise RuntimeError("System-wide UI Automation currently requires Windows.")
        try:
            from pywinauto import Desktop
        except ImportError as exc:
            raise RuntimeError("Install the optional pywinauto dependency to enable desktop automation.") from exc
        return Desktop(backend="uia")

    def execute(self, intent: str, parameters: Dict[str, Any], context: Dict[str, Any]) -> PluginExecutionResult:
        try:
            desktop = self._desktop()
            if intent == "click_ui_control" and parameters.get("expected_window_title") and parameters.get("expected_process_id"):
                window = desktop.window(
                    title=parameters["expected_window_title"],
                    process=int(parameters["expected_process_id"]),
                ).wrapper_object()
            else:
                window = desktop.get_active()
            if window is None or not window.exists():
                return PluginExecutionResult(success=False, spoken_response="I could not identify the target application window.")

            if intent == "get_active_window":
                title = (window.window_text() or "").strip()
                return PluginExecutionResult(
                    success=True,
                    spoken_response=f"The active window is {title or 'an unnamed application window'}.",
                    data={"title": title, "process_id": window.process_id()},
                )

            if intent == "click_ui_control":
                name = str(parameters.get("control_name", "")).strip()
                if not name or len(name) > 160:
                    return PluginExecutionResult(success=False, spoken_response="Tell me the exact accessible name of the control.")
                control_type = str(parameters.get("control_type", "Button")).strip()
                if control_type not in {"Button", "MenuItem", "TabItem", "CheckBox", "RadioButton", "Hyperlink", "ListItem"}:
                    return PluginExecutionResult(success=False, spoken_response="That control type is not supported for voice clicking.")
                # Match exact accessible names only; no coordinate guessing or fuzzy target selection.
                matches = window.descendants(title=name, control_type=control_type)
                if len(matches) != 1:
                    return PluginExecutionResult(
                        success=False,
                        spoken_response=f"I found {len(matches)} matching {control_type} controls named {name}. Please clarify the control.",
                    )
                matches[0].click_input()
                return PluginExecutionResult(success=True, spoken_response=f"Clicked {name} in {window.window_text() or 'the active window'}.")

            if intent == "type_text":
                text = parameters.get("text")
                if not isinstance(text, str) or not text or len(text) > 4000:
                    return PluginExecutionResult(success=False, spoken_response="Provide text of 1 to 4000 characters.")
                # Refuse typing into password fields when UI Automation exposes that metadata.
                try:
                    focused = [
                        control for control in window.descendants()
                        if bool(getattr(control.element_info, "has_keyboard_focus", False))
                    ]
                    if len(focused) != 1:
                        return PluginExecutionResult(success=False, spoken_response="I cannot verify the focused control, so I did not type.")
                    element = focused[0].element_info
                    if element.control_type == "Edit":
                        auto_id = (element.automation_id or "").lower()
                        name = (element.name or "").lower()
                        if "password" in auto_id or "password" in name:
                            return PluginExecutionResult(success=False, spoken_response="For safety, I will not type into a password field.")
                except Exception:
                    # Focus metadata can be unavailable for custom controls; fail closed.
                    return PluginExecutionResult(success=False, spoken_response="I cannot verify the focused control, so I did not type.")
                from pywinauto.keyboard import send_keys
                escaped = "".join("{{}" if char == "{" else "{}}" if char == "}" else char for char in text)
                send_keys(escaped, with_spaces=True, pause=0.002)
                return PluginExecutionResult(success=True, spoken_response="Typed the requested text into the focused control.")

            if intent == "press_hotkey":
                requested = str(parameters.get("keys", "")).strip().lower().replace(" ", "")
                # Normalize the documented "page up/down" spellings.
                requested = requested.replace("pageup", "page up").replace("pagedown", "page down")
                sequence = SAFE_HOTKEYS.get(requested)
                if sequence is None:
                    return PluginExecutionResult(success=False, spoken_response="That keyboard shortcut is not on the safe shortcut list.")
                from pywinauto.keyboard import send_keys
                send_keys(sequence)
                return PluginExecutionResult(success=True, spoken_response=f"Pressed {requested}.")

            return PluginExecutionResult(success=False, spoken_response="Unsupported desktop automation action.")
        except Exception as exc:
            logger.exception("Desktop automation failed")
            return PluginExecutionResult(
                success=False,
                spoken_response="I could not safely interact with the active application.",
                error=str(exc),
            )
