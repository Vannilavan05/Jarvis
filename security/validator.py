"""Security Validator for Jarvis.
Enforces path safety, directory isolation, filename sanitization, and command injection defense.
"""

import os
import re
from pathlib import Path
from typing import List, Optional, Tuple


class SecurityValidator:
    """Provides path containment and argument validation."""

    # Disallowed characters in file names
    INVALID_FILENAME_CHARS = re.compile(r'[<>:"/\\|?*\x00-\x1f]')

    @staticmethod
    def is_safe_path(target_path: Path, allowed_directories: List[Path]) -> Tuple[bool, str]:
        """Validates that target_path resolves strictly within one of the allowed directories.

        Protects against path traversal attacks and unauthorized directory access.
        """
        try:
            resolved_target = target_path.resolve()
        except Exception as e:
            return False, f"Failed to resolve path: {str(e)}"

        # Check against every allowed directory
        for allowed in allowed_directories:
            try:
                resolved_allowed = allowed.resolve()
                # Ensure target is equal to or sub-path of allowed
                try:
                    resolved_target.relative_to(resolved_allowed)
                    return True, "Path is authorized."
                except ValueError:
                    continue
            except Exception:
                continue

        return False, f"Access denied: Path '{target_path}' is outside authorized directories."

    @staticmethod
    def sanitize_filename(filename: str) -> str:
        """Sanitizes a filename to prevent path injection and invalid characters."""
        # Strip directory components
        base = os.path.basename(filename).strip()
        # Remove dangerous characters
        cleaned = SecurityValidator.INVALID_FILENAME_CHARS.sub("_", base)
        # Avoid reserved names or blank
        if not cleaned or cleaned in {".", ".."}:
            cleaned = "unnamed_file.txt"
        return cleaned

    @staticmethod
    def is_safe_argument(argument: str) -> bool:
        """Ensures command argument contains no shell metacharacters."""
        dangerous_patterns = [";", "&", "|", "`", "$", "(", ")", ">", "<", "\n", "\r"]
        return not any(ch in argument for ch in dangerous_patterns)
