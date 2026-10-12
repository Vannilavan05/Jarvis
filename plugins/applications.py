"""Application launcher with installed-app discovery and no shell-string execution.

The configured aliases remain supported, but opening an app no longer requires
adding its name to a fixed allowlist: Jarvis searches Start Menu shortcuts, App Paths,
and executable search paths for installed applications.
"""
import logging
import os
import platform
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

try:
    from security.permissions import PermissionLevel
except (ImportError, ValueError):
    from jarvis.security.permissions import PermissionLevel
from .base import BasePlugin, PluginExecutionResult

logger = logging.getLogger(__name__)
LaunchTarget = Union[str, Path]

class ApplicationPlugin(BasePlugin):
    """Discover and launch installed desktop apps without per-app plugin code."""

    DEFAULT_APPROVED_APPS = {
        "chrome": "chrome", "browser": "chrome", "notepad": "notepad.exe",
        "calculator": "calc.exe", "calc": "calc.exe", "explorer": "explorer.exe",
        "code": "code.cmd", "cmd": "cmd.exe", "terminal": "wt.exe",
    }

    @property
    def plugin_id(self) -> str:
        return "applications"

    @property
    def name(self) -> str:
        return "Application Launcher"

    @property
    def version(self) -> str:
        return "2.0.0"

    @property
    def supported_intents(self) -> List[str]:
        return ["open_application", "close_application"]

    @property
    def required_permissions(self) -> Dict[str, PermissionLevel]:
        return {"open_application": PermissionLevel.STANDARD, "close_application": PermissionLevel.DESTRUCTIVE}

    @staticmethod
    def _normalize(value: str) -> str:
        return re.sub(r"[^a-z0-9]+", "", value.casefold())

    def _start_menu_shortcuts(self) -> List[Path]:
        if platform.system() != "Windows":
            return []
        roots = []
        for env_name in ("PROGRAMDATA", "APPDATA"):
            value = os.environ.get(env_name)
            if value:
                root = Path(value) / "Microsoft" / "Windows" / "Start Menu" / "Programs"
                if root.is_dir():
                    roots.append(root)
        found = []
        for root in roots:
            try:
                found.extend(path for path in root.rglob("*.lnk") if path.is_file())
                found.extend(path for path in root.rglob("*.url") if path.is_file())
            except OSError:
                logger.debug("Could not fully scan Start Menu folder %s", root, exc_info=True)
        return found

    def _app_path_registry(self, name: str) -> Optional[str]:
        if platform.system() != "Windows":
            return None
        try:
            import winreg
            keys = (
                (winreg.HKEY_CURRENT_USER, rf"Software\Microsoft\Windows\CurrentVersion\App Paths\{name}"),
                (winreg.HKEY_LOCAL_MACHINE, rf"Software\Microsoft\Windows\CurrentVersion\App Paths\{name}"),
            )
            for hive, key_path in keys:
                try:
                    with winreg.OpenKey(hive, key_path) as key:
                        value, _ = winreg.QueryValueEx(key, None)
                        if isinstance(value, str) and Path(value.strip('"')).exists():
                            return value.strip('"')
                except OSError:
                    continue
        except ImportError:
            return None
        return None

    def _discover_installed_app(self, app_name: str) -> Optional[LaunchTarget]:
        """Resolve an exact app name first; only accept partial matches when unique."""
        requested = self._normalize(app_name)
        if not requested or len(requested) > 100:
            return None

        shortcuts = self._start_menu_shortcuts()
        exact = [p for p in shortcuts if self._normalize(p.stem) == requested]
        if len(exact) == 1:
            return exact[0]
        if len(exact) > 1:
            return None

        partial = [p for p in shortcuts if requested in self._normalize(p.stem)]
        if len(partial) == 1:
            return partial[0]
        if len(partial) > 1:
            return None

        # Windows App Paths entries allow discovery of installed Win32 applications.
        for candidate in (app_name, app_name + ".exe"):
            resolved = self._app_path_registry(candidate)
            if resolved:
                return resolved

        # Executables already exposed on PATH are also valid installed apps.
        for candidate in (app_name, app_name + ".exe", app_name + ".cmd", app_name + ".bat"):
            resolved = shutil.which(candidate)
            if resolved:
                return resolved
        return None

    def _resolve_launch_target(self, app_name: str, approved_map: Dict[str, Any]) -> Optional[LaunchTarget]:
        # Preserve user-defined aliases, but resolve their values as executable names/paths.
        configured = approved_map.get(app_name)
        candidates = []
        if isinstance(configured, list):
            candidates.extend(str(item) for item in configured if item)
        elif isinstance(configured, str) and configured:
            candidates.append(configured)
        candidates.append(app_name)

        for candidate in candidates:
            discovered = self._discover_installed_app(candidate)
            if discovered:
                return discovered
            if Path(candidate).is_file():
                return candidate
        return None

    def execute(self, intent: str, parameters: Dict[str, Any], context: Dict[str, Any]) -> PluginExecutionResult:
        app_name = str(parameters.get("application", "")).strip()
        if not app_name:
            return PluginExecutionResult(success=False, spoken_response="No application name specified.", error="Missing 'application' parameter")

        if len(app_name) > 100 or any(ch in app_name for ch in ("\x00", "\n", "\r")):
            return PluginExecutionResult(success=False, spoken_response="That application name is invalid.")

        app_key = app_name.casefold()
        settings = context.get("settings")
        approved_map = settings.get("approved_applications", self.DEFAULT_APPROVED_APPS) if settings else self.DEFAULT_APPROVED_APPS

        if intent == "open_application":
            target = self._resolve_launch_target(app_key, approved_map)
            if target is None:
                return PluginExecutionResult(
                    success=False,
                    spoken_response=f"I could not find {app_name} among installed applications or the approved application list.",
                    error=f"Installed application not found: {app_name}",
                )
            try:
                if platform.system() == "Windows" and str(target).lower().endswith((".lnk", ".url")):
                    os.startfile(str(target))
                elif platform.system() == "Windows" and isinstance(target, Path):
                    os.startfile(str(target))
                else:
                    subprocess.Popen([str(target)], shell=False)
                return PluginExecutionResult(success=True, spoken_response=f"Opening {app_name}.", data={"application": app_name, "target": str(target)})
            except (OSError, ValueError) as exc:
                logger.warning("Failed to launch discovered app %s: %s", app_name, exc)
                return PluginExecutionResult(success=False, spoken_response=f"Windows could not open {app_name}.", error=str(exc))

        if intent == "close_application":
            if platform.system() != "Windows":
                return PluginExecutionResult(success=False, spoken_response="Closing applications by process name is currently supported only on Windows.")
            # Termination is intentionally restricted to a simple executable name.
            process_target = app_key if app_key.endswith(".exe") else f"{app_key}.exe"
            if not re.fullmatch(r"[a-z0-9_. -]{1,120}\.exe", process_target):
                return PluginExecutionResult(success=False, spoken_response="That process name is not safe to terminate.")
            try:
                res = subprocess.run(["taskkill", "/F", "/IM", process_target], capture_output=True, text=True, shell=False, timeout=10)
                if res.returncode == 0:
                    return PluginExecutionResult(success=True, spoken_response=f"Closed {app_name}.", data={"application": app_name})
                return PluginExecutionResult(success=False, spoken_response=f"Could not find a running process for {app_name}.", error=res.stderr.strip())
            except (OSError, subprocess.TimeoutExpired) as exc:
                return PluginExecutionResult(success=False, spoken_response=f"Error terminating {app_name}.", error=str(exc))

        return PluginExecutionResult(success=False, spoken_response="Unsupported application action.")
