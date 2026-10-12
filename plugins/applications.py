"""Application Launcher Plugin for Jarvis.
Opens and terminates approved desktop applications safely.
"""

import logging
import os
import subprocess
from typing import Any, Dict, List

try:
    from security.permissions import PermissionLevel
except (ImportError, ValueError):
    from jarvis.security.permissions import PermissionLevel
from .base import BasePlugin, PluginExecutionResult

logger = logging.getLogger(__name__)


class ApplicationPlugin(BasePlugin):
    """Safely launches and manages approved desktop applications."""

    DEFAULT_APPROVED_APPS = {
        "chrome": ["start", "chrome"],  # Handled via os.startfile on Windows where possible
        "browser": ["start", "chrome"],
        "notepad": ["notepad.exe"],
        "calculator": ["calc.exe"],
        "calc": ["calc.exe"],
        "explorer": ["explorer.exe"],
        "code": ["code.cmd"],
        "cmd": ["cmd.exe"],
        "terminal": ["wt.exe"],
    }

    @property
    def plugin_id(self) -> str:
        return "applications"

    @property
    def name(self) -> str:
        return "Application Launcher"

    @property
    def version(self) -> str:
        return "1.0.0"

    @property
    def supported_intents(self) -> List[str]:
        return ["open_application", "close_application"]

    @property
    def required_permissions(self) -> Dict[str, PermissionLevel]:
        return {
            "open_application": PermissionLevel.STANDARD,
            "close_application": PermissionLevel.DESTRUCTIVE,
        }

    def execute(
        self, intent: str, parameters: Dict[str, Any], context: Dict[str, Any]
    ) -> PluginExecutionResult:
        app_name = parameters.get("application", "").strip().lower()
        if not app_name:
            return PluginExecutionResult(
                success=False,
                spoken_response="No application name specified.",
                error="Missing 'application' parameter",
            )

        settings = context.get("settings")
        approved_map = (
            settings.get("approved_applications", self.DEFAULT_APPROVED_APPS)
            if settings
            else self.DEFAULT_APPROVED_APPS
        )

        if app_name not in approved_map:
            # Check for partial match
            match = None
            for key in approved_map.keys():
                if app_name in key or key in app_name:
                    match = key
                    break
            if match:
                app_name = match
            else:
                return PluginExecutionResult(
                    success=False,
                    spoken_response=f"Application {app_name} is not in the approved application list.",
                    error=f"Unapproved application '{app_name}'",
                )

        target_cmd = approved_map[app_name]

        if intent == "open_application":
            try:
                # Safe execution without raw shell string concatenation
                if isinstance(target_cmd, list):
                    cmd_to_run = target_cmd[0]
                else:
                    cmd_to_run = str(target_cmd)

                # On Windows, try os.startfile for seamless desktop app launches
                if hasattr(os, "startfile") and not cmd_to_run.endswith(".exe"):
                    try:
                        os.startfile(cmd_to_run)
                    except Exception:
                        subprocess.Popen([cmd_to_run], shell=False)
                else:
                    subprocess.Popen([cmd_to_run], shell=False)

                return PluginExecutionResult(
                    success=True,
                    spoken_response=f"Opening {app_name}.",
                    data={"application": app_name},
                )
            except Exception as e:
                logger.error(f"Failed to launch application {app_name}: {e}")
                return PluginExecutionResult(
                    success=False,
                    spoken_response=f"Failed to launch {app_name}.",
                    error=str(e),
                )

        elif intent == "close_application":
            try:
                # Use taskkill safely on Windows
                process_target = f"{app_name}.exe"
                res = subprocess.run(
                    ["taskkill", "/F", "/IM", process_target],
                    capture_output=True,
                    text=True,
                    shell=False,
                )
                if res.returncode == 0:
                    return PluginExecutionResult(
                        success=True,
                        spoken_response=f"Closed {app_name}.",
                        data={"application": app_name},
                    )
                else:
                    return PluginExecutionResult(
                        success=False,
                        spoken_response=f"Could not find a running process for {app_name}.",
                        error=res.stderr.strip(),
                    )
            except Exception as e:
                return PluginExecutionResult(
                    success=False,
                    spoken_response=f"Error terminating {app_name}.",
                    error=str(e),
                )

        return PluginExecutionResult(success=False, spoken_response="Unsupported application action.")
