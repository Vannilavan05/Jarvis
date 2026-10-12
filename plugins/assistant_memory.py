"""Voice-accessible local memory and automation status plugin."""
from typing import Any, Dict, List

try:
    from security.permissions import PermissionLevel
except (ImportError, ValueError):
    from jarvis.security.permissions import PermissionLevel
from .base import BasePlugin, PluginExecutionResult


class AssistantMemoryPlugin(BasePlugin):
    @property
    def plugin_id(self) -> str:
        return "assistant_memory"

    @property
    def name(self) -> str:
        return "Assistant Memory and Progress"

    @property
    def version(self) -> str:
        return "1.0.0"

    @property
    def supported_intents(self) -> List[str]:
        return ["remember_fact", "recall_memories", "get_task_status", "get_recent_history"]

    @property
    def required_permissions(self) -> Dict[str, PermissionLevel]:
        return {
            "remember_fact": PermissionLevel.STANDARD,
            "recall_memories": PermissionLevel.READ_ONLY,
            "get_task_status": PermissionLevel.READ_ONLY,
            "get_recent_history": PermissionLevel.READ_ONLY,
        }

    def execute(self, intent: str, parameters: Dict[str, Any], context: Dict[str, Any]) -> PluginExecutionResult:
        memory = context.get("assistant_memory")
        if memory is None:
            return PluginExecutionResult(False, "Local memory is not available right now.")
        try:
            if intent == "remember_fact":
                key = parameters.get("memory_key", "user_note")
                value = parameters.get("memory_value")
                if not isinstance(value, str) or not value.strip():
                    return PluginExecutionResult(False, "Tell me exactly what you want me to remember.")
                memory.remember(key, value)
                return PluginExecutionResult(True, f"I saved that locally as {key}.")
            if intent == "recall_memories":
                items = memory.recall(limit=20)
                if not items:
                    return PluginExecutionResult(True, "I do not have any saved memories yet.", data=[])
                summary = "; ".join(f"{item['memory_key']}: {item['memory_value']}" for item in items)
                return PluginExecutionResult(True, "Here is what I remember: " + summary, data=items)
            if intent == "get_task_status":
                tasks = memory.pending_tasks()
                if not tasks:
                    recent = memory.recent_tasks(limit=1)
                    if not recent:
                        return PluginExecutionResult(True, "I have no saved automation tasks yet.", data=[])
                    item = recent[0]
                    return PluginExecutionResult(True, f"Your latest task, {item['title']}, is {item['status']}.", data=recent)
                summary = "; ".join(f"task {item['id']}: {item['title']} — {item['status']}, step {item['current_step'] + 1}" for item in tasks[:10])
                return PluginExecutionResult(True, "Here is your task progress: " + summary, data=tasks[:10])
            if intent == "get_recent_history":
                messages = memory.recent_messages("default", limit=10)
                if not messages:
                    return PluginExecutionResult(True, "There is no saved conversation history yet.", data=[])
                summary = " | ".join(f"{m['role']}: {m['content']}" for m in messages)
                return PluginExecutionResult(True, "Recent conversation history: " + summary, data=messages)
            return PluginExecutionResult(False, "Unsupported memory operation.")
        except (ValueError, TypeError) as exc:
            return PluginExecutionResult(False, "I could not save or retrieve that memory safely.", error=str(exc))
