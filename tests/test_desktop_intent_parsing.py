"""Tests that local LLM parsing cannot invent desktop targets or dictated text."""
from ai.local_llm import Interpretation
from execution.command_registry import CommandRegistry
from nlp.intent_parser import IntentParser


class FakeRouter:
    def __init__(self, result):
        self.result = result

    def interpret(self, _text):
        return self.result


def make_interpretation(intent, parameters):
    return Interpretation(
        mode="action_request",
        intent=intent,
        parameters=parameters,
        response=None,
        needs_clarification=False,
    )


def test_type_text_uses_exact_spoken_text_and_preserves_case():
    router = FakeRouter(make_interpretation("type_text", {"text": "model-invented text"}))
    parser = IntentParser(predictor=object(), llm_router=router, command_registry=CommandRegistry())

    result = parser._try_local_llm("Jarvis, type Hello World", "type hello world")

    assert result is not None
    assert result.intent == "type_text"
    assert result.parameters["text"] == "Hello World"


def test_llm_cannot_invent_application_name():
    router = FakeRouter(make_interpretation("open_application", {"application": "Notepad"}))
    parser = IntentParser(predictor=object(), llm_router=router, command_registry=CommandRegistry())

    result = parser._try_local_llm("Jarvis, open Paint", "open paint")

    assert result is not None
    assert result.is_ambiguous is True
    assert "exact application" in result.clarification_prompt
