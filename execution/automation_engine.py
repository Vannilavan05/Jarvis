"""Resumable, allowlisted multi-step automation for Jarvis.

The engine never executes shell/code directly. Every step is routed back through
CommandDispatcher, so the existing registry, permission checks, confirmation,
plugin handlers, and audit logging remain authoritative.
"""
from __future__ import annotations

import logging
from typing import Any, Callable, Dict, Iterable, List

try:
    from nlp.intent_parser import CommandObject
except (ImportError, ValueError):
    from jarvis.nlp.intent_parser import CommandObject

logger = logging.getLogger(__name__)

# Only routine, bounded actions can be composed into a plan. Destructive/system
# control, model training, confirmation, and nested automation are never plan steps.
AUTOMATION_INTENTS = {
    "open_application", "get_active_window", "click_ui_control", "type_text",
    "press_hotkey", "web_search", "read_file", "list_files", "system_info",
}
MAX_STEPS = 10


class AutomationEngine:
    def __init__(self, memory, command_registry):
        self.memory = memory
        self.registry = command_registry

    def validate_steps(self, steps: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not isinstance(steps, list) or not 1 <= len(steps) <= MAX_STEPS:
            raise ValueError(f"An automation plan must contain 1 to {MAX_STEPS} steps.")
        validated = []
        for index, step in enumerate(steps):
            if not isinstance(step, dict) or set(step) - {"intent", "parameters"}:
                raise ValueError(f"Automation step {index + 1} has an invalid shape.")
            intent, params = step.get("intent"), step.get("parameters", {})
            if intent not in AUTOMATION_INTENTS:
                raise ValueError(f"Automation step {index + 1} uses a restricted or unsupported action.")
            if not self.registry.get(intent):
                raise ValueError(f"Automation step {index + 1} is not registered.")
            if not isinstance(params, dict):
                raise ValueError(f"Automation step {index + 1} parameters must be an object.")
            metadata = self.registry.get(intent)
            missing = [name for name in metadata.required_parameters if params.get(name) in (None, "", [])]
            if missing:
                raise ValueError(f"Automation step {index + 1} is missing: {', '.join(missing)}.")
            # Prevent a planner from using desktop text entry to send commands to shells.
            if intent == "type_text" and (not isinstance(params.get("text"), str) or len(params["text"]) > 4000):
                raise ValueError(f"Automation step {index + 1} has invalid dictated text.")
            validated.append({"intent": intent, "parameters": dict(params)})
        return validated

    def start(self, title: str, steps: Iterable[Dict[str, Any]], dispatch: Callable[[CommandObject], Any]) -> Dict[str, Any]:
        validated = self.validate_steps(list(steps))
        task_id = self.memory.create_task(title, validated)
        return self._run(task_id, dispatch)

    def resume(self, dispatch: Callable[[CommandObject], Any], confirmed_pending_step: bool = False, task_id: int = None) -> Dict[str, Any]:
        """Resume a specific paused task or the newest interrupted task."""
        candidates = self.memory.pending_tasks()
        task = next((item for item in candidates if (task_id is None or item["id"] == task_id) and item["status"] in {"waiting_confirmation", "running", "planned"}), None)
        if task is None:
            return {"success": True, "message": "There is no paused automation task to resume."}
        full = self.memory.get_task(task["id"])
        index = int(full["current_step"])
        if confirmed_pending_step and index < len(full["steps"]):
            self.memory.update_step(full["id"], index, "completed", "Completed after spoken confirmation.")
            index += 1
            self.memory.update_task(full["id"], "running", current_step=index)
        return self._run(full["id"], dispatch)

    def _run(self, task_id: int, dispatch: Callable[[CommandObject], Any]) -> Dict[str, Any]:
        task = self.memory.get_task(task_id)
        if task is None:
            return {"success": False, "message": "Automation task was not found."}
        steps = task["steps"]
        index = int(task["current_step"])
        self.memory.update_task(task_id, "running", current_step=index)
        while index < len(steps):
            step = steps[index]
            if step["status"] == "completed":
                index += 1
                continue
            self.memory.update_step(task_id, index, "running")
            command = CommandObject(
                intent=step["intent"], parameters=step["parameters"], confidence=1.0,
                is_ambiguous=False, raw_text=f"Approved automation task {task_id}, step {index + 1}",
                model_version=None,
            )
            try:
                response = dispatch(command)
            except Exception as exc:
                logger.exception("Automation task %s failed at step %s", task_id, index + 1)
                self.memory.update_step(task_id, index, "failed", str(exc))
                self.memory.update_task(task_id, "failed", current_step=index, error=str(exc))
                return {"success": False, "task_id": task_id, "message": f"Automation stopped at step {index + 1} due to an error."}
            if getattr(response, "requires_confirmation", False):
                self.memory.update_step(task_id, index, "waiting_confirmation", response.spoken_feedback)
                self.memory.update_task(task_id, "waiting_confirmation", current_step=index)
                return {"success": True, "task_id": task_id, "waiting_confirmation": True,
                        "message": f"Automation is paused at step {index + 1} for confirmation. {response.spoken_feedback}"}
            if not getattr(response, "success", False):
                reason = getattr(response, "spoken_feedback", "Step failed.")
                self.memory.update_step(task_id, index, "failed", reason)
                self.memory.update_task(task_id, "failed", current_step=index, error=reason)
                return {"success": False, "task_id": task_id, "message": f"Automation stopped at step {index + 1}: {reason}"}
            self.memory.update_step(task_id, index, "completed", getattr(response, "spoken_feedback", "Completed."))
            index += 1
            self.memory.update_task(task_id, "running", current_step=index)
            task = self.memory.get_task(task_id)
            steps = task["steps"]
        self.memory.update_task(task_id, "completed", current_step=len(steps))
        return {"success": True, "task_id": task_id, "message": f"Automation completed all {len(steps)} steps."}
