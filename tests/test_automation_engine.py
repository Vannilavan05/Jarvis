from types import SimpleNamespace

import pytest

from execution.automation_engine import AutomationEngine
from execution.command_registry import CommandRegistry
from storage.assistant_memory import AssistantMemory
from storage.database import DatabaseManager


@pytest.fixture
def engine(tmp_path):
    memory = AssistantMemory(DatabaseManager(tmp_path / "automation.db"))
    return AutomationEngine(memory, CommandRegistry()), memory


def test_automation_runs_steps_in_order_and_persists_completion(engine):
    automation, memory = engine
    seen = []

    def dispatch(command):
        seen.append((command.intent, command.parameters))
        return SimpleNamespace(success=True, spoken_feedback="Done", requires_confirmation=False)

    result = automation.start("Open Chrome then check active window", [
        {"intent": "open_application", "parameters": {"application": "Chrome"}},
        {"intent": "get_active_window", "parameters": {}},
    ], dispatch)
    task = memory.get_task(result["task_id"])
    assert result["success"] is True
    assert task["status"] == "completed"
    assert [step["status"] for step in task["steps"]] == ["completed", "completed"]
    assert [item[0] for item in seen] == ["open_application", "get_active_window"]


def test_automation_pauses_and_resumes_after_confirmation(engine):
    automation, memory = engine
    calls = {"count": 0}

    def dispatch(command):
        calls["count"] += 1
        if command.intent == "click_ui_control":
            return SimpleNamespace(success=True, spoken_feedback="Please confirm click", requires_confirmation=True)
        return SimpleNamespace(success=True, spoken_feedback="Typed text", requires_confirmation=False)

    result = automation.start("Click Save then type note", [
        {"intent": "click_ui_control", "parameters": {"control_name": "Save"}},
        {"intent": "type_text", "parameters": {"text": "Draft"}},
    ], dispatch)
    assert result["waiting_confirmation"] is True
    task = memory.get_task(result["task_id"])
    assert task["status"] == "waiting_confirmation"
    resumed = automation.resume(dispatch, confirmed_pending_step=True, task_id=result["task_id"])
    assert resumed["success"] is True
    assert memory.get_task(result["task_id"])["status"] == "completed"


def test_automation_rejects_destructive_and_nested_steps(engine):
    automation, _ = engine
    with pytest.raises(ValueError):
        automation.validate_steps([{"intent": "delete_file", "parameters": {"filename": "notes.txt"}}])
    with pytest.raises(ValueError):
        automation.validate_steps([{"intent": "run_automation", "parameters": {"steps": []}}])
