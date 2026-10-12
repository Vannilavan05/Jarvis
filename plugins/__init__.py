"""Jarvis Plugins Subsystem.
"""

from .applications import ApplicationPlugin
from .base import BasePlugin, PluginExecutionResult
from .custom_commands import CustomCommandsPlugin
from .files import FilePlugin
from .model_management import ModelManagementPlugin
from .system_controls import SystemControlsPlugin
from .system_info import SystemInfoPlugin
from .web_search import WebSearchPlugin

__all__ = [
    "BasePlugin",
    "PluginExecutionResult",
    "ApplicationPlugin",
    "FilePlugin",
    "WebSearchPlugin",
    "SystemInfoPlugin",
    "SystemControlsPlugin",
    "ModelManagementPlugin",
    "CustomCommandsPlugin",
]
