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
        (r"^(?:confirm(?:\s+deletion)?|yes\s+confirm|authorize|proceed)$", "confirm_action"),

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
    ):
        self.predictor = predictor or IntentPredictor(confidence_threshold=confidence_threshold)
        self.confidence_mgr = ConfidenceManager(threshold=confidence_threshold)

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

        # 3. Neural inference via PyTorch IntentPredictor
        pred = self.predictor.predict(normalized)

        if not pred.intent:
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
