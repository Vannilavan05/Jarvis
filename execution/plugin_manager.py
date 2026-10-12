"""Plugin Manager for Jarvis.
Handles discovery, registration, allowlisting, and routing to execution plugins.
"""
import logging
from typing import Dict, List, Optional

try:
    from plugins.applications import ApplicationPlugin
    from plugins.assistant_memory import AssistantMemoryPlugin
    from plugins.base import BasePlugin
    from plugins.custom_commands import CustomCommandsPlugin
    from plugins.desktop_automation import DesktopAutomationPlugin
    from plugins.files import FilePlugin
    from plugins.model_management import ModelManagementPlugin
    from plugins.system_controls import SystemControlsPlugin
    from plugins.system_info import SystemInfoPlugin
    from plugins.web_search import WebSearchPlugin
except (ImportError, ValueError):
    from jarvis.plugins.applications import ApplicationPlugin
    from jarvis.plugins.assistant_memory import AssistantMemoryPlugin
    from jarvis.plugins.base import BasePlugin
    from jarvis.plugins.custom_commands import CustomCommandsPlugin
    from jarvis.plugins.desktop_automation import DesktopAutomationPlugin
    from jarvis.plugins.files import FilePlugin
    from jarvis.plugins.model_management import ModelManagementPlugin
    from jarvis.plugins.system_controls import SystemControlsPlugin
    from jarvis.plugins.system_info import SystemInfoPlugin
    from jarvis.plugins.web_search import WebSearchPlugin

logger = logging.getLogger(__name__)


class PluginManager:
    """Coordinates modular execution plugins."""

    def __init__(self, enabled_plugin_ids: Optional[List[str]] = None):
        self._plugins: Dict[str, BasePlugin] = {}
        self._intent_map: Dict[str, BasePlugin] = {}
        self._enabled_ids: set = set(enabled_plugin_ids) if enabled_plugin_ids is not None else set()
        self._register_default_plugins()

    def register_plugin(self, plugin: BasePlugin):
        self._plugins[plugin.plugin_id] = plugin
        if not self._enabled_ids or plugin.plugin_id in self._enabled_ids:
            for intent in plugin.supported_intents:
                self._intent_map[intent] = plugin
        logger.info("Registered plugin '%s' (id: %s, version: %s)", plugin.name, plugin.plugin_id, plugin.version)

    def get_plugin_for_intent(self, intent: str) -> Optional[BasePlugin]:
        return self._intent_map.get(intent)

    def is_plugin_enabled(self, plugin_id: str) -> bool:
        return not self._enabled_ids or plugin_id in self._enabled_ids

    def get_all_plugins(self) -> List[BasePlugin]:
        return list(self._plugins.values())

    def _register_default_plugins(self):
        self.register_plugin(ApplicationPlugin())
        self.register_plugin(AssistantMemoryPlugin())
        self.register_plugin(DesktopAutomationPlugin())
        self.register_plugin(FilePlugin())
        self.register_plugin(WebSearchPlugin())
        self.register_plugin(SystemInfoPlugin())
        self.register_plugin(SystemControlsPlugin())
        self.register_plugin(ModelManagementPlugin())
        self.register_plugin(CustomCommandsPlugin())
