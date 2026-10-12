from nlp.intent_parser import IntentParser


class NeverCalledPredictor:
    def predict(self, text):
        raise AssertionError("Deterministic memory and compound-command routes should run before prediction")


def parser_with_fake_llm(fake_router=None):
    from execution.command_registry import CommandRegistry
    return IntentParser(predictor=NeverCalledPredictor(), llm_router=fake_router, command_registry=CommandRegistry())


def test_remember_fact_preserves_user_text():
    cmd = parser_with_fake_llm().parse("Jarvis, remember that I am learning AWS")
    assert cmd.intent == "remember_fact"
    assert cmd.parameters["memory_value"] == "I am learning AWS"


def test_task_status_is_deterministic():
    cmd = parser_with_fake_llm().parse("Jarvis, what is pending?")
    assert cmd.intent == "get_task_status"


def test_compound_request_does_not_fall_back_to_single_legacy_intent():
    cmd = parser_with_fake_llm().parse("Jarvis, open Chrome then search for Python tutorials")
    assert cmd.is_ambiguous is True
    assert "automation plan" in cmd.clarification_prompt
