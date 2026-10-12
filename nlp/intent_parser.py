"""Intent Parser and Structured Command Constructor for Jarvis.
Combines priority/emergency pattern matching, PyTorch intent classification,
and parameter extraction into a unified command lifecycle.
"""

import logging
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

try:
    from ai.predictor import IntentPredictor, PredictionResult
except (ImportError, ValueError):
    from jarvis.ai.predictor import IntentPredictor, PredictionResult
from .confidence import ConfidenceManager
from .entity_extractor import EntityExtractor

logger = logging.getLogger(__name__)


@dataclass
class CommandObject:
    """Structured representation of a parsed user command."""
    intent: str
    parameters: Dict[str, Any]
    confidence: float
    is_emergency: bool = False
    is_ambiguous: bool = False
    clarification_prompt: Optional[str] = None
    raw_text: str = ""
    model_version: Optional[int] = None


class IntentParser:
    """Parses spoken text into validated CommandObjects."""

    # Emergency and deterministic priority commands that bypass neural inference
    PRIORITY_RULES = [
        # Emergency stop & listening state
        (r"^(?:stop\s+listening|shut\s+down|exit|terminate|emergency\s+stop)$", "stop_listening"),
        (r"^(?:pause\s+listening|pause|standby)$", "pause_listening"),
        (r"^(?:resume\s+listening|resume|continue)$", "resume_listening"),
        (r"^(?:cancel(?:\s+the)?\s+current\s+command|cancel\s+pending\s+actions?|cancel|abort|return\s+to\s+idle)$", "cancel_action"),

        # Confirmation phrases
        (r"^(?:confirm|yes[ ,]+confirm|authorize|proceed)$", "confirm_action"),

        # Voice training mode triggers
        (r"^(?:enter\s+training\s+mode|start\s+training\s+mode)$", "start_training"),
        (r"^(?:train\s+the\s+model|retrain(?:\s+neural\s+network)?)$", "train_model"),
        (r"^(?:evaluate\s+the\s+model|check\s+model\s+accuracy)$", "evaluate_model"),
        (r"^(?:tell\s+me\s+the\s+current\s+model\s+version|what\s+is\s+the\s+model\s+version)$", "get_model_version"),
        (r"^(?:restore\s+the\s+previous\s+model|rollback\s+model)$", "rollback_model"),
        (r"^(?:review\s+my\s+training\s+examples|show\s+training\s+examples)$", "review_examples"),
        (r"^(?:delete\s+(?:that|last)\s+training\s+example|remove\s+last\s+example)$", "delete_last_example"),

        # Direct web search triggers
        (r"^(?:search(?:\s+the\s+web)?(?:\s+online)?(?:\s+for)?|google|look\s+up)\s+(.+)$", "web_search"),

        # Direct file triggers
        (r"^(?:delete|remove)\s+(?:file|the\s+file)\s+(.+)$", "delete_file"),
        (r"^(?:read|show(?:\s+content\s+of)?)\s+(?:file|the\s+file)\s+(.+)$", "read_file"),
        (r"^(?:create|make)\s+(?:file|a\s+new\s+file(?:\s+called)?)\s+(.+)$", "create_file"),
        (r"^(?:list\s+files|show\s+files)$", "list_files"),
    ]

    def __init__(
        self,
        predictor: Optional[IntentPredictor] = None,
        confidence_threshold: float = 0.65,
        llm_router=None,
        command_registry=None,
        assistant_memory=None,
    ):
        self.predictor = predictor or IntentPredictor(confidence_threshold=confidence_threshold)
        self.confidence_mgr = ConfidenceManager(threshold=confidence_threshold)
        self.llm_router = llm_router
        self.command_registry = command_registry
        self.assistant_memory = assistant_memory

    def normalize(self, text: str) -> str:
        """Strips assistant wake name and conversational prefixes."""
        cleaned = text.strip().lower()
        # Remove wake word occurrences at start or end
        cleaned = re.sub(r"^(?:jarvis|hey\s+jarvis|okay\s+jarvis)\s*[,:]?\s*", "", cleaned)
        cleaned = re.sub(r"\s*[,:]?\s*(?:jarvis)$", "", cleaned)
        # Remove politeness fillers
        cleaned = re.sub(r"^(?:please|could\s+you|can\s+you|would\s+you)\s+", "", cleaned)
        return cleaned.strip()

    def parse(self, raw_text: str) -> CommandObject:
        """Parses speech text into a structured CommandObject."""
        normalized = self.normalize(raw_text)
        if not normalized:
            return CommandObject(
                intent="unknown",
                parameters={},
                confidence=0.0,
                is_ambiguous=True,
                clarification_prompt="I heard you, but the command was empty. How may I assist you?",
                raw_text=raw_text,
            )

        # 1. Check priority / emergency deterministic rules
        for pattern, intent in self.PRIORITY_RULES:
            if re.search(pattern, normalized):
                is_emergency = intent in {"stop_listening", "cancel_action"}
                params = EntityExtractor.extract(raw_text, intent)
                return CommandObject(
                    intent=intent,
                    parameters=params,
                    confidence=1.0,
                    is_emergency=is_emergency,
                    is_ambiguous=False,
                    clarification_prompt=None,
                    raw_text=raw_text,
                    model_version=None,
                )

        # 2. Check for Training Example syntax ("When I say X, do Y")
        if normalized.startswith("when i say ") or " add a training example" in normalized:
            params = EntityExtractor.extract(raw_text, "add_example")
            return CommandObject(
                intent="add_example",
                parameters=params,
                confidence=1.0,
                is_emergency=False,
                is_ambiguous=False,
                clarification_prompt=None,
                raw_text=raw_text,
                model_version=None,
            )

        # 3. Deterministic desktop commands. The trained classifier predates these
        # intents, so route explicit, unambiguous desktop requests before neural inference.
        original_command = re.sub(
            r"^(?:hey\s+jarvis|okay\s+jarvis|jarvis)\s*[, :]?\s*",
            "",
            raw_text.strip(),
            flags=re.IGNORECASE,
        )
        original_command = re.sub(
            r"^(?:please|could\s+you|can\s+you|would\s+you)\s+",
            "",
            original_command,
            flags=re.IGNORECASE,
        ).strip()

        if re.fullmatch(r"(?:what(?:'s| is) (?:the )?active window|which (?:app|application|window) is active|identify active window|what app am i using)", normalized):
            return CommandObject(
                intent="get_active_window", parameters={}, confidence=1.0,
                raw_text=raw_text, model_version=None,
            )

        typed_match = re.match(r"^(?:type|enter|write|input|insert)\s+(.+)$", original_command, re.IGNORECASE)
        if typed_match:
            dictated_text = typed_match.group(1).strip()
            if dictated_text:
                return CommandObject(
                    intent="type_text", parameters={"text": dictated_text}, confidence=1.0,
                    raw_text=raw_text, model_version=None,
                )

        hotkey_match = re.fullmatch(r"(?:press|hit|use)\s+(?:the\s+)?(.+)", normalized)
        if hotkey_match:
            shortcut = hotkey_match.group(1).strip().replace(" ", "")
            shortcut = shortcut.replace("pageup", "page up").replace("pagedown", "page down")
            if shortcut in {
                "ctrl+a", "ctrl+c", "ctrl+v", "ctrl+x", "ctrl+z", "ctrl+y",
                "ctrl+s", "ctrl+f", "ctrl+p", "ctrl+w", "ctrl+tab",
                "ctrl+shift+tab", "alt+tab", "enter", "escape", "tab",
                "shift+tab", "home", "end", "page up", "page down",
            }:
                return CommandObject(
                    intent="press_hotkey", parameters={"keys": shortcut}, confidence=1.0,
                    raw_text=raw_text, model_version=None,
                )
            return CommandObject(
                intent="unknown", parameters={}, confidence=0.0, is_ambiguous=True,
                clarification_prompt="That keyboard shortcut is not on Jarvis's safe shortcut list.",
                raw_text=raw_text, model_version=None,
            )

        click_match = re.fullmatch(
            r"(?:click|select)\s+(?:the\s+)?(.+?)(?:\s+(button|menu item|tab|checkbox|radio button|link|list item))?",
            original_command,
            flags=re.IGNORECASE,
        )
        if click_match:
            control_name = click_match.group(1).strip()
            suffix = (click_match.group(2) or "button").lower()
            control_type = {
                "button": "Button", "menu item": "MenuItem", "tab": "TabItem",
                "checkbox": "CheckBox", "radio button": "RadioButton",
                "link": "Hyperlink", "list item": "ListItem",
            }[suffix]
            if control_name:
                return CommandObject(
                    intent="click_ui_control",
                    parameters={"control_name": control_name, "control_type": control_type},
                    confidence=1.0, raw_text=raw_text, model_version=None,
                )

        open_match = re.fullmatch(r"(?:open|launch|start|run)\s+(?:the\s+)?(.+)", original_command, flags=re.IGNORECASE)
        if open_match:
            application = open_match.group(1).strip()
            if application:
                return CommandObject(
                    intent="open_application", parameters={"application": application},
                    confidence=1.0, raw_text=raw_text, model_version=None,
                )

        # Persistent memory and progress commands are deterministic and voice-first.
        remember_match = re.fullmatch(r"remember(?: that)?\s+(.+)", original_command, re.IGNORECASE)
        if remember_match:
            fact = remember_match.group(1).strip()
            key_slug = re.sub(r"[^a-z0-9]+", "_", fact.lower()).strip("_")[:72] or "user_note"
            return CommandObject(intent="remember_fact", parameters={"memory_key": "note_" + key_slug, "memory_value": fact}, confidence=1.0, raw_text=raw_text, model_version=None)
        if re.fullmatch(r"(?:what do you remember|show my saved memories|recall my memories)", normalized):
            return CommandObject(intent="recall_memories", parameters={}, confidence=1.0, raw_text=raw_text, model_version=None)
        if re.fullmatch(r"(?:what (?:is|are) pending|what have we been working on|show task status|show my progress|what is the status of my tasks)", normalized):
            return CommandObject(intent="get_task_status", parameters={}, confidence=1.0, raw_text=raw_text, model_version=None)
        if re.fullmatch(r"(?:what did we discuss recently|show recent conversation|recall our recent conversation)", normalized):
            return CommandObject(intent="get_recent_history", parameters={}, confidence=1.0, raw_text=raw_text, model_version=None)
        if re.fullmatch(r"(?:continue|resume) (?:the )?(?:previous|last|pending) (?:automation|task)", normalized):
            return CommandObject(intent="resume_automation", parameters={}, confidence=1.0, raw_text=raw_text, model_version=None)

        # Explicit multi-step requests must go through the local planner; never silently
        # reduce a compound request to one guessed action from the legacy classifier.
        if re.search(r"\b(?:automate|then|after that|and then|step by step)\b", normalized):
            planned = self._try_local_llm(raw_text, normalized)
            if planned is not None and planned.intent == "run_automation":
                return planned
            return CommandObject(intent="unknown", parameters={}, confidence=0.0, is_ambiguous=True,
                                 clarification_prompt="I could not safely build a complete automation plan. Please state the steps explicitly.", raw_text=raw_text)

        # 4. Neural inference via PyTorch IntentPredictor
        pred = self.predictor.predict(normalized)

        if not pred.intent:
            llm_command = self._try_local_llm(raw_text, normalized)
            if llm_command is not None:
                return llm_command
            return CommandObject(
                intent="unknown",
                parameters={},
                confidence=0.0,
                is_ambiguous=True,
                clarification_prompt="I am not sure how to process that request. Could you rephrase?",
                raw_text=raw_text,
                model_version=pred.model_version,
            )

        # Evaluate confidence
        is_confident, clarification = self.confidence_mgr.evaluate_confidence(
            intent=pred.intent,
            confidence=pred.confidence,
            top_candidates=pred.top_candidates,
        )

        params = EntityExtractor.extract(raw_text, pred.intent)

        # Use the local LLM only when the existing PyTorch model is uncertain.
        if not is_confident:
            llm_command = self._try_local_llm(raw_text, normalized)
            if llm_command is not None:
                return llm_command

        # Validate mandatory parameters for specific intents
        if is_confident:
            if pred.intent in {"open_application", "close_application"} and not params.get("application"):
                is_confident = False
                clarification = "Which application would you like me to open?"
            elif pred.intent in {"read_file", "create_file", "delete_file"} and not params.get("filename"):
                is_confident = False
                clarification = "Which file would you like me to target?"
            elif pred.intent == "web_search" and not params.get("query"):
                is_confident = False
                clarification = "What would you like me to search for on the web?"

        return CommandObject(
            intent=pred.intent,
            parameters=params,
            confidence=pred.confidence,
            is_emergency=False,
            is_ambiguous=not is_confident,
            clarification_prompt=clarification,
            raw_text=raw_text,
            model_version=pred.model_version,
        )

    def _try_local_llm(self, raw_text: str, normalized: str) -> Optional[CommandObject]:
        """Ask the configured local LLM only after deterministic/PyTorch parsing is uncertain."""
        if self.llm_router is None:
            return None
        try:
            allowed = self.command_registry.all_intents() if self.command_registry is not None else {}
            if self.assistant_memory is not None:
                history = self.assistant_memory.recent_messages("default", limit=8)
                result = self.llm_router.interpret(normalized, history=history)
            else:
                result = self.llm_router.interpret(normalized)
            # Do not let the model invent dictated text or an application/control target.
            if result.mode == "action_request" and result.intent == "type_text":
                spoken_text = raw_text.strip()
                spoken_text = re.sub(r"^(?:hey\s+jarvis|okay\s+jarvis|jarvis)\s*[, :]?\s*", "", spoken_text, flags=re.IGNORECASE)
                spoken_text = re.sub(r"^(?:please|could\s+you|can\s+you|would\s+you)\s+", "", spoken_text, flags=re.IGNORECASE)
                dictated = re.match(r"^(?:type|enter|write|input|insert)\s+(.+)$", spoken_text, re.IGNORECASE)
                if not dictated:
                    return CommandObject(
                        intent="unknown", parameters={}, confidence=0.0, is_ambiguous=True,
                        clarification_prompt="Tell me exactly what text you want me to type.",
                        raw_text=raw_text,
                    )
                exact_text = dictated.group(1).strip()
                safe_parameters = dict(result.parameters)
                safe_parameters["text"] = exact_text
                result = type(result)(
                    mode=result.mode, intent=result.intent, parameters=safe_parameters,
                    response=result.response, needs_clarification=result.needs_clarification,
                )
            if result.mode == "action_request" and result.intent in {"open_application", "click_ui_control"}:
                key = "application" if result.intent == "open_application" else "control_name"
                target = result.parameters.get(key)
                if not isinstance(target, str) or not target.strip() or target.casefold() not in normalized.casefold():
                    return CommandObject(
                        intent="unknown", parameters={}, confidence=0.0, is_ambiguous=True,
                        clarification_prompt="Please state the exact application or control name you want me to use.",
                        raw_text=raw_text,
                    )
            if result.mode == "action_request" and result.intent == "run_automation":
                # Every user-facing target/value must be grounded in the spoken request.
                for step in result.parameters.get("steps", []):
                    for key in ("application", "control_name", "text", "query", "filename"):
                        value = step.get("parameters", {}).get(key)
                        if isinstance(value, str) and value.strip() and value.casefold() not in normalized.casefold():
                            return CommandObject(intent="unknown", parameters={}, confidence=0.0, is_ambiguous=True,
                                clarification_prompt=f"I could not verify the requested {key.replace('_', ' ')} from your words. Please state it explicitly.", raw_text=raw_text)
            if result.mode == "conversation":
                return CommandObject(
                    intent="conversation", parameters={}, confidence=1.0,
                    is_ambiguous=False, clarification_prompt=result.response,
                    raw_text=raw_text, model_version=None,
                )
            if result.mode == "clarification":
                return CommandObject(
                    intent="unknown", parameters={}, confidence=0.0,
                    is_ambiguous=True, clarification_prompt=result.response,
                    raw_text=raw_text, model_version=None,
                )
            if self.command_registry is not None and result.intent not in allowed:
                return None
            meta = allowed.get(result.intent) if allowed else None
            if meta is not None:
                missing = [key for key in meta.required_parameters if not result.parameters.get(key)]
                if missing:
                    return CommandObject(
                        intent="unknown", parameters={}, confidence=0.0, is_ambiguous=True,
                        clarification_prompt=f"I need more information to {result.intent.replace('_', ' ')}. Please specify {', '.join(missing)}.",
                        raw_text=raw_text,
                    )
            return CommandObject(
                intent=result.intent, parameters=result.parameters, confidence=0.75,
                is_ambiguous=False, raw_text=raw_text, model_version=None,
            )
        except Exception as exc:
            logger.warning("Local LLM unavailable or rejected output; using safe clarification: %s", exc)
            return None
