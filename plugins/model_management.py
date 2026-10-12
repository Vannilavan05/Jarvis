"""Model Management Plugin for Jarvis.
Handles spoken model training, evaluation, rollback, and training example management.
"""

import logging
from typing import Any, Dict, List

try:
    from ai.dataset import DatasetManager
    from ai.evaluator import ModelEvaluator
    from ai.tokenizer import JarvisTokenizer
    from ai.trainer import ModelTrainer
    from security.permissions import PermissionLevel
except (ImportError, ValueError):
    from jarvis.ai.dataset import DatasetManager
    from jarvis.ai.evaluator import ModelEvaluator
    from jarvis.ai.tokenizer import JarvisTokenizer
    from jarvis.ai.trainer import ModelTrainer
    from jarvis.security.permissions import PermissionLevel
from .base import BasePlugin, PluginExecutionResult

logger = logging.getLogger(__name__)


class ModelManagementPlugin(BasePlugin):
    """Executes model training, evaluation, checkpoints, and dataset updates."""

    @property
    def plugin_id(self) -> str:
        return "model_management"

    @property
    def name(self) -> str:
        return "Model Management"

    @property
    def version(self) -> str:
        return "1.0.0"

    @property
    def supported_intents(self) -> List[str]:
        return [
            "start_training",
            "add_example",
            "review_examples",
            "delete_last_example",
            "train_model",
            "evaluate_model",
            "get_model_version",
            "rollback_model",
        ]

    @property
    def required_permissions(self) -> Dict[str, PermissionLevel]:
        return {
            "start_training": PermissionLevel.STANDARD,
            "add_example": PermissionLevel.STANDARD,
            "review_examples": PermissionLevel.READ_ONLY,
            "delete_last_example": PermissionLevel.DESTRUCTIVE,
            "train_model": PermissionLevel.STANDARD,
            "evaluate_model": PermissionLevel.READ_ONLY,
            "get_model_version": PermissionLevel.READ_ONLY,
            "rollback_model": PermissionLevel.DESTRUCTIVE,
        }

    def execute(
        self, intent: str, parameters: Dict[str, Any], context: Dict[str, Any]
    ) -> PluginExecutionResult:
        training_store = context.get("training_store")
        model_registry = context.get("model_registry")
        predictor = context.get("predictor")

        if intent == "start_training":
            return PluginExecutionResult(
                success=True,
                spoken_response="Training mode activated. Tell me the command you want me to learn, or say train the model.",
                data={"mode": "training"},
            )

        elif intent == "add_example":
            if not training_store:
                return PluginExecutionResult(
                    success=False, spoken_response="Training store unavailable.", error="No training store"
                )

            phrase = parameters.get("phrase", "")
            action = parameters.get("action", "")

            # If parameters weren't explicitly matched as "when I say...", extract from raw or arguments
            if not phrase and parameters.get("raw_text"):
                raw = parameters.get("raw_text", "")
                phrase = raw

            if not phrase:
                return PluginExecutionResult(
                    success=False,
                    spoken_response="Please state the command you would like me to learn.",
                    error="Missing phrase",
                )

            # Map action phrase to intent if provided, else deduce from action keywords
            target_intent = "open_application"
            if "search" in action:
                target_intent = "web_search"
            elif "file" in action:
                target_intent = "read_file"
            elif "volume" in action or "sound" in action:
                target_intent = "system_control"

            ok, msg, record_id = training_store.add_example(
                text=phrase,
                intent=target_intent,
                entities={"action": action} if action else {},
                approved=True,
            )
            if ok:
                resp = f"I understood the action as {action or target_intent}. Training example saved."
                return PluginExecutionResult(success=True, spoken_response=resp, data={"id": record_id})
            else:
                return PluginExecutionResult(success=False, spoken_response=f"Failed to add example: {msg}", error=msg)

        elif intent == "review_examples":
            if not training_store:
                return PluginExecutionResult(success=False, spoken_response="Training store unavailable.")
            count = training_store.count_examples()
            last = training_store.get_last_example()
            if count == 0:
                return PluginExecutionResult(
                    success=True, spoken_response="There are currently no training examples in the dataset."
                )
            last_text = last["text"] if last else "none"
            last_intent = last["intent"] if last else "none"
            resp = f"You have {count} approved training examples. The most recent example is '{last_text}' mapped to '{last_intent}'."
            return PluginExecutionResult(success=True, spoken_response=resp, data={"count": count, "last": last})

        elif intent == "delete_last_example":
            if not training_store:
                return PluginExecutionResult(success=False, spoken_response="Training store unavailable.")
            ok, msg = training_store.delete_last_example()
            return PluginExecutionResult(success=ok, spoken_response=msg, error=None if ok else msg)

        elif intent == "train_model":
            if not training_store or not model_registry:
                return PluginExecutionResult(success=False, spoken_response="Model subsystems unavailable.")

            examples = training_store.get_all_examples(approved_only=True)
            trainer = ModelTrainer(registry=model_registry)
            result = trainer.train(examples, epochs=30)

            if result.success:
                if predictor:
                    predictor.reload()
                return PluginExecutionResult(
                    success=True, spoken_response=result.summary_message, data={"version": result.version}
                )
            else:
                return PluginExecutionResult(
                    success=False, spoken_response=result.summary_message, error=result.summary_message
                )

        elif intent == "evaluate_model":
            if not model_registry or not predictor:
                return PluginExecutionResult(success=False, spoken_response="Model evaluation unavailable.")

            if predictor.model is None:
                return PluginExecutionResult(
                    success=False,
                    spoken_response="No trained model is currently active. Say 'train the model' first.",
                )

            # Evaluate against current dataset
            examples = training_store.get_all_examples(approved_only=True) if training_store else []

            tokenizer = predictor.tokenizer or JarvisTokenizer()
            splits = DatasetManager.prepare_datasets(examples, tokenizer)
            metrics = ModelEvaluator.evaluate(
                model=predictor.model,
                dataset=splits.test_dataset or splits.train_dataset,
                idx_to_label=predictor.idx_to_label,
                device=predictor.device,
            )
            return PluginExecutionResult(
                success=True, spoken_response=metrics.summary, data={"metrics": metrics}
            )

        elif intent == "get_model_version":
            if not model_registry:
                return PluginExecutionResult(success=False, spoken_response="Model registry unavailable.")
            v = model_registry.get_active_version()
            if v is not None:
                resp = f"The active neural model is version {v}."
            else:
                resp = "There is no active trained model version."
            return PluginExecutionResult(success=True, spoken_response=resp, data={"version": v})

        elif intent == "rollback_model":
            if not model_registry:
                return PluginExecutionResult(success=False, spoken_response="Model registry unavailable.")
            ok, msg, prev_v = model_registry.rollback()
            if ok and predictor:
                predictor.reload()
            return PluginExecutionResult(success=ok, spoken_response=msg, data={"version": prev_v}, error=None if ok else msg)

        return PluginExecutionResult(success=False, spoken_response="Unsupported model command.")
