import pytest

from storage.assistant_memory import AssistantMemory
from storage.database import DatabaseManager


def make_memory(tmp_path):
    return AssistantMemory(DatabaseManager(tmp_path / "jarvis-test.db"))


def test_remember_upserts_and_recalls_locally(tmp_path):
    memory = make_memory(tmp_path)
    memory.remember("learning_goal", "Learn AWS networking")
    memory.remember("learning_goal", "Learn AWS networking and IAM")
    items = memory.recall("learning_goal")
    assert len(items) == 1
    assert items[0]["memory_value"] == "Learn AWS networking and IAM"


def test_conversation_history_is_ordered_and_limited(tmp_path):
    memory = make_memory(tmp_path)
    memory.save_message("default", "user", "Hello Jarvis")
    memory.save_message("default", "assistant", "Hello")
    history = memory.recent_messages("default", limit=2)
    assert [item["role"] for item in history] == ["user", "assistant"]


def test_task_and_steps_survive_new_memory_instance(tmp_path):
    db = DatabaseManager(tmp_path / "jarvis-test.db")
    memory = AssistantMemory(db)
    task_id = memory.create_task("Open browser", [{"intent": "open_application", "parameters": {"application": "Chrome"}}])
    memory.update_step(task_id, 0, "completed", "Opened Chrome")
    memory.update_task(task_id, "completed", current_step=1)
    reloaded = AssistantMemory(DatabaseManager(tmp_path / "jarvis-test.db"))
    task = reloaded.get_task(task_id)
    assert task["status"] == "completed"
    assert task["steps"][0]["status"] == "completed"


def test_memory_rejects_empty_values(tmp_path):
    memory = make_memory(tmp_path)
    with pytest.raises(ValueError):
        memory.remember("", "value")
