"""Wake-Word Detection Engine for Jarvis.
Provides local offline wake-word spotting and command detachment.
"""

import re
from typing import List, Optional, Tuple


class WakeWordDetector:
    """Detects configurable wake words locally in speech transcripts."""

    def __init__(self, wake_word: str = "jarvis", aliases: Optional[List[str]] = None):
        self.wake_word = wake_word.strip().lower()
        self.aliases = [a.strip().lower() for a in (aliases or ["hey jarvis", "okay jarvis", "jarvis"])]
        if self.wake_word not in self.aliases:
            self.aliases.append(self.wake_word)

    def set_wake_word(self, wake_word: str):
        self.wake_word = wake_word.strip().lower()
        if self.wake_word not in self.aliases:
            self.aliases.append(self.wake_word)

    def detect_in_text(self, text: str) -> Tuple[bool, str]:
        """Checks if text contains the wake word.

        Returns (is_detected, remainder_command_text).
        For example: "jarvis open chrome" -> (True, "open chrome").
        """
        cleaned = text.strip().lower()
        if not cleaned:
            return False, ""

        for alias in sorted(self.aliases, key=len, reverse=True):
            # Check if alias appears at the start
            pattern = rf"^{re.escape(alias)}\b\s*[,:]?\s*(.*)$"
            match = re.match(pattern, cleaned)
            if match:
                remainder = match.group(1).strip()
                return True, remainder

            # Or contains isolated wake word
            standalone_pattern = rf"\b{re.escape(alias)}\b"
            if re.search(standalone_pattern, cleaned):
                # Strip out the wake word
                remainder = re.sub(standalone_pattern, "", cleaned).strip()
                remainder = re.sub(r"^[,\s:]+", "", remainder)
                return True, remainder

        return False, ""
