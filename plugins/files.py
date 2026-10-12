"""File Management Plugin for Jarvis.
Performs sandboxed, validated file reading, creation, listing, and deletion.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List

try:
    from config import DATA_DIR
    from security.permissions import PermissionLevel
    from security.validator import SecurityValidator
except (ImportError, ValueError):
    from jarvis.config import DATA_DIR
    from jarvis.security.permissions import PermissionLevel
    from jarvis.security.validator import SecurityValidator
from .base import BasePlugin, PluginExecutionResult

logger = logging.getLogger(__name__)


class FilePlugin(BasePlugin):
    """Handles sandboxed file system operations with path containment checks."""

    @property
    def plugin_id(self) -> str:
        return "files"

    @property
    def name(self) -> str:
        return "File Manager"

    @property
    def version(self) -> str:
        return "1.0.0"

    @property
    def supported_intents(self) -> List[str]:
        return ["read_file", "create_file", "delete_file", "list_files"]

    @property
    def required_permissions(self) -> Dict[str, PermissionLevel]:
        return {
            "read_file": PermissionLevel.READ_ONLY,
            "list_files": PermissionLevel.READ_ONLY,
            "create_file": PermissionLevel.FILE_MODIFICATION,
            "delete_file": PermissionLevel.DESTRUCTIVE,
        }

    def _resolve_target(self, filename: str, allowed_dirs: List[Path]) -> Path:
        """Resolves target file path within primary data directory if not an absolute path."""
        p = Path(filename)
        if not p.is_absolute():
            primary_dir = allowed_dirs[0] if allowed_dirs else DATA_DIR
            p = primary_dir / SecurityValidator.sanitize_filename(filename)
        return p

    def execute(
        self, intent: str, parameters: Dict[str, Any], context: Dict[str, Any]
    ) -> PluginExecutionResult:
        settings = context.get("settings")
        allowed_dirs = (
            settings.get_allowed_directories()
            if hasattr(settings, "get_allowed_directories")
            else [DATA_DIR]
        )

        filename = parameters.get("filename", "").strip()

        if intent == "list_files":
            all_files: List[str] = []
            for d in allowed_dirs:
                if d.exists() and d.is_dir():
                    try:
                        for item in d.iterdir():
                            if item.is_file():
                                all_files.append(item.name)
                    except Exception:
                        continue
            count = len(all_files)
            if count == 0:
                return PluginExecutionResult(
                    success=True,
                    spoken_response="There are no files in your authorized directories.",
                    data={"files": []},
                )
            preview = ", ".join(all_files[:5])
            response_msg = f"Found {count} files: {preview}" if count <= 5 else f"Found {count} files including: {preview}"
            return PluginExecutionResult(
                success=True,
                spoken_response=response_msg,
                data={"files": all_files},
            )

        if not filename:
            return PluginExecutionResult(
                success=False,
                spoken_response="Please specify the file name.",
                error="Missing filename",
            )

        target_path = self._resolve_target(filename, allowed_dirs)
        is_safe, reason = SecurityValidator.is_safe_path(target_path, allowed_dirs)
        if not is_safe:
            return PluginExecutionResult(
                success=False,
                spoken_response="Access denied. The requested file path is outside authorized directories.",
                error=reason,
            )

        if intent == "read_file":
            if not target_path.exists():
                return PluginExecutionResult(
                    success=False,
                    spoken_response=f"The file {target_path.name} was not found.",
                    error="File not found",
                )
            try:
                content = target_path.read_text(encoding="utf-8")
                # Keep spoken response concise (first 250 characters)
                preview = content[:250].strip()
                summary = f"Reading {target_path.name}: {preview}" if preview else f"File {target_path.name} is empty."
                return PluginExecutionResult(
                    success=True,
                    spoken_response=summary,
                    data={"filename": str(target_path), "content": content},
                )
            except Exception as e:
                return PluginExecutionResult(
                    success=False,
                    spoken_response=f"Failed to read file: {str(e)}",
                    error=str(e),
                )

        elif intent == "create_file":
            try:
                content = parameters.get("content", f"Created by Jarvis on {target_path.name}\n")
                target_path.parent.mkdir(parents=True, exist_ok=True)
                # Atomic write via temp file
                tmp = target_path.with_suffix(f"{target_path.suffix}.tmp")
                tmp.write_text(content, encoding="utf-8")
                tmp.replace(target_path)
                return PluginExecutionResult(
                    success=True,
                    spoken_response=f"Successfully created file {target_path.name}.",
                    data={"filename": str(target_path)},
                )
            except Exception as e:
                return PluginExecutionResult(
                    success=False,
                    spoken_response=f"Could not create file {target_path.name}.",
                    error=str(e),
                )

        elif intent == "delete_file":
            if not target_path.exists():
                return PluginExecutionResult(
                    success=False,
                    spoken_response=f"The file {target_path.name} does not exist.",
                    error="File not found",
                )
            try:
                target_path.unlink()
                return PluginExecutionResult(
                    success=True,
                    spoken_response=f"File {target_path.name} has been deleted.",
                    data={"filename": str(target_path)},
                )
            except Exception as e:
                return PluginExecutionResult(
                    success=False,
                    spoken_response=f"Failed to delete {target_path.name}.",
                    error=str(e),
                )

        return PluginExecutionResult(success=False, spoken_response="Unknown file operation.")
