"""Web Search Plugin for Jarvis.
Performs URL-encoded web queries via the system default browser.
"""

import logging
import urllib.parse
import webbrowser
from typing import Any, Dict, List

try:
    from security.permissions import PermissionLevel
except (ImportError, ValueError):
    from jarvis.security.permissions import PermissionLevel
from .base import BasePlugin, PluginExecutionResult

logger = logging.getLogger(__name__)


class WebSearchPlugin(BasePlugin):
    """Executes validated web search requests."""

    SEARCH_ENGINE_URL = "https://www.google.com/search?q="

    @property
    def plugin_id(self) -> str:
        return "web_search"

    @property
    def name(self) -> str:
        return "Web Search"

    @property
    def version(self) -> str:
        return "1.0.0"

    @property
    def supported_intents(self) -> List[str]:
        return ["web_search"]

    @property
    def required_permissions(self) -> Dict[str, PermissionLevel]:
        return {"web_search": PermissionLevel.STANDARD}

    def execute(
        self, intent: str, parameters: Dict[str, Any], context: Dict[str, Any]
    ) -> PluginExecutionResult:
        query = parameters.get("query", "").strip()
        if not query:
            return PluginExecutionResult(
                success=False,
                spoken_response="What would you like me to search for?",
                error="Empty search query",
            )

        encoded_query = urllib.parse.quote_plus(query)
        full_url = f"{self.SEARCH_ENGINE_URL}{encoded_query}"

        try:
            # Safe browser open without shell execution
            webbrowser.open(full_url)
            return PluginExecutionResult(
                success=True,
                spoken_response=f"Searching the web for {query}.",
                data={"query": query, "url": full_url},
            )
        except Exception as e:
            logger.error(f"Failed to open browser for search: {e}")
            return PluginExecutionResult(
                success=False,
                spoken_response="Could not initiate web search.",
                error=str(e),
            )
