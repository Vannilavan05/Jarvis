"""Entity and Parameter Extractor for Jarvis.
Extracts structured parameters from natural language command utterances.
"""

import re
from typing import Any, Dict


class EntityExtractor:
    """Extracts entities such as application names, file paths, and search queries."""

    # Common known applications
    APP_ALIASES = {
        "google chrome": "chrome",
        "chrome": "chrome",
        "browser": "chrome",
        "web browser": "chrome",
        "notepad": "notepad",
        "text editor": "notepad",
        "calculator": "calculator",
        "calc": "calculator",
        "file explorer": "explorer",
        "files": "explorer",
        "explorer": "explorer",
        "visual studio code": "code",
        "vs code": "code",
        "vscode": "code",
        "terminal": "terminal",
        "command prompt": "cmd",
        "cmd": "cmd",
    }

    @classmethod
    def extract(cls, text: str, intent: str) -> Dict[str, Any]:
        """Extracts parameters for the given intent from text."""
        params: Dict[str, Any] = {}
        cleaned = text.strip()
        lower = cleaned.lower()

        if intent in {"open_application", "close_application"}:
            params.update(cls._extract_application(lower))
        elif intent == "web_search":
            params.update(cls._extract_search_query(cleaned))
        elif intent in {"read_file", "create_file", "delete_file"}:
            params.update(cls._extract_filename(cleaned))
        elif intent == "system_info":
            params.update(cls._extract_system_metric(lower))
        elif intent == "system_control":
            params.update(cls._extract_control_action(lower))
        elif intent == "add_example":
            params.update(cls._extract_training_pair(cleaned))

        return params

    @classmethod
    def _extract_application(cls, text: str) -> Dict[str, str]:
        for alias, normalized in cls.APP_ALIASES.items():
            if alias in text:
                return {"application": normalized}

        # Fallback regex e.g. "open <app>"
        match = re.search(r"(?:open|launch|start|close|terminate)\s+([a-zA-Z0-9_\-\s]+)", text)
        if match:
            app_raw = match.group(1).strip()
            # Clean common words like 'the' or 'my'
            app_clean = re.sub(r"^(the|my)\s+", "", app_raw).strip()
            return {"application": app_clean}
        return {}

    @classmethod
    def _extract_search_query(cls, text: str) -> Dict[str, str]:
        # Strip search triggers
        patterns = [
            r"^(?:search(?:\s+the\s+web)?(?:\s+online)?(?:\s+for)?)\s+(.*)$",
            r"^(?:google(?:\s+for)?)\s+(.*)$",
            r"^(?:look\s+up)\s+(.*)$",
            r"^(?:find\s+information\s+on)\s+(.*)$",
        ]
        for pat in patterns:
            match = re.search(pat, text, re.IGNORECASE)
            if match:
                query = match.group(1).strip()
                if query:
                    return {"query": query}
        return {"query": text.strip()}

    @classmethod
    def _extract_filename(cls, text: str) -> Dict[str, str]:
        # Match filenames with extension or after keywords
        match = re.search(r"([a-zA-Z0-9_\-.]+\.[a-zA-Z0-9]+)", text)
        if match:
            return {"filename": match.group(1)}

        # Match "file <name>"
        match_named = re.search(r"(?:file|called|named)\s+([a-zA-Z0-9_\-]+)", text, re.IGNORECASE)
        if match_named:
            name = match_named.group(1).strip()
            return {"filename": f"{name}.txt"}

        return {}

    @classmethod
    def _extract_system_metric(cls, text: str) -> Dict[str, str]:
        if "cpu" in text or "processor" in text:
            return {"metric": "cpu"}
        elif "ram" in text or "memory" in text:
            return {"metric": "memory"}
        elif "disk" in text or "storage" in text or "drive" in text:
            return {"metric": "disk"}
        elif "battery" in text or "power" in text:
            return {"metric": "battery"}
        elif "network" in text or "ip" in text or "wifi" in text:
            return {"metric": "network"}
        return {"metric": "all"}

    @classmethod
    def _extract_control_action(cls, text: str) -> Dict[str, str]:
        if "up" in text or "increase" in text or "raise" in text or "louder" in text:
            return {"action": "volume_up"}
        elif "down" in text or "decrease" in text or "lower" in text or "quieter" in text:
            return {"action": "volume_down"}
        elif "unmute" in text:
            return {"action": "unmute"}
        elif "mute" in text or "silence" in text:
            return {"action": "mute"}
        elif "lock" in text:
            return {"action": "lock"}
        return {"action": "unknown"}

    @classmethod
    def _extract_training_pair(cls, text: str) -> Dict[str, str]:
        # Match pattern: "when I say <phrase>, <action>"
        match = re.search(r"when\s+i\s+say\s+(.+?),\s*(?:then\s+)?(.+)", text, re.IGNORECASE)
        if match:
            return {
                "phrase": match.group(1).strip(),
                "action": match.group(2).strip(),
            }
        return {}
