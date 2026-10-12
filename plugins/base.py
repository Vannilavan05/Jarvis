"""Base Plugin Interface for Jarvis Execution Engine.
Defines plugin contracts, intent declarations, and execution results.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

try:
    from security.permissions import PermissionLevel
except (ImportError, ValueError):
    from jarvis.security.permissions import PermissionLevel


@dataclass
class PluginExecutionResult:
    """Standardized result returned by all plugin executions."""
    success: bool
    spoken_response: str
    data: Optional[Any] = None
    error: Optional[str] = None


class BasePlugin(ABC):
    """Abstract base class for all modular execution plugins in Jarvis."""

    @property
    @abstractmethod
    def plugin_id(self) -> str:
        """Unique machine identifier for the plugin."""
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable plugin name."""
        pass

    @property
    @abstractmethod
    def version(self) -> str:
        """Semantic version of the plugin."""
        pass

    @property
    @abstractmethod
    def supported_intents(self) -> List[str]:
        """List of intent names handled by this plugin."""
        pass

    @property
    @abstractmethod
    def required_permissions(self) -> Dict[str, PermissionLevel]:
        """Mapping from intent name to minimum PermissionLevel required."""
        pass

    @abstractmethod
    def execute(
        self, intent: str, parameters: Dict[str, Any], context: Dict[str, Any]
    ) -> PluginExecutionResult:
        """Executes the approved and validated intent."""
        pass
