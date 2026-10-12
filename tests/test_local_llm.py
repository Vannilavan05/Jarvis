import pytest

from ai.local_llm import (
    LocalLLMRouter,
    OllamaLocalRuntime,
    validate_interpretation,
)

ALLOWED = {"open_application", "read_file", "web_search", "list_files"}

def valid(mode="action_request", intent="open_application", parameters=None, response=None, needs_clarification=False):
    return {
        "mode": mode,
        "intent": intent,
        "parameters": parameters or {},
        "response": response,
        "needs_clarification": needs_clarification,
    }

def test_valid_action_is_structured():
    parsed = validate_interpretation(valid(parameters={"application": "notepad"}), ALLOWED)
    assert parsed.intent == "open_application"
    assert parsed.parameters == {"application": "notepad"}

@pytest.mark.parametrize("intent", [
    "confirm_action", "cancel_action", "stop_listening", "shutdown_system",
    "reboot_system", "delete_file", "close_application", "rollback_model",
    "delete_last_example", "custom_command", "train_model",
])
def test_restricted_intents_are_rejected(intent):
    with pytest.raises(ValueError):
        validate_interpretation(valid(intent=intent), ALLOWED | {intent})

def test_unknown_intent_is_rejected():
    with pytest.raises(ValueError):
        validate_interpretation(valid(intent="run_shell_command"), ALLOWED)

def test_extra_fields_are_rejected():
    data = valid()
    data["execute_now"] = True
    with pytest.raises(ValueError):
        validate_interpretation(data, ALLOWED)

def test_conversation_cannot_smuggle_action():
    with pytest.raises(ValueError):
        validate_interpretation(valid(mode="conversation", intent=None, parameters={"path": "x"}, response="Done"), ALLOWED)

def test_conversation_response_is_validated():
    parsed = validate_interpretation(valid(mode="conversation", intent=None, response="Hello"), ALLOWED)
    assert parsed.mode == "conversation"
    assert parsed.response == "Hello"

def test_remote_endpoint_is_rejected():
    with pytest.raises(ValueError):
        OllamaLocalRuntime(base_url="https://example.com")

def test_non_loopback_http_is_rejected():
    with pytest.raises(ValueError):
        OllamaLocalRuntime(base_url="http://192.168.1.10:11434")

def test_router_rejects_empty_user_text():
    router = LocalLLMRouter(OllamaLocalRuntime(), ALLOWED)
    with pytest.raises(ValueError):
        router.interpret("   ")
