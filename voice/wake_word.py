"""Wake-Word Detection Engine for Jarvis.
Provides local offline wake-word spotting and command detachment.
"""

import re
from difflib import SequenceMatcher
from typing import List, Optional, Tuple


class WakeWordDetector:
    """Detects configurable wake words locally in speech transcripts."""

    # Minimum similarity (0-1) for a mis-transcribed wake word at the start of an utterance
    FUZZY_THRESHOLD = 0.8

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

        ordered_aliases = sorted(self.aliases, key=len, reverse=True)

        for alias in ordered_aliases:
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

        # Speech models often mishear the wake word slightly ("jervis", "jarvus"); tolerate that at the
        # start of the utterance only, so ordinary words later in a sentence never trigger it.
        words = cleaned.split()
        for alias in ordered_aliases:
            n = len(alias.split())
            if len(words) < n:
                continue
            heard = " ".join(words[:n]).strip(",:;.!?")
            if SequenceMatcher(None, heard, alias).ratio() >= self.FUZZY_THRESHOLD:
                remainder = " ".join(words[n:]).strip()
                remainder = re.sub(r"^[,\s:]+", "", remainder)
                return True, remainder

        return False, ""
