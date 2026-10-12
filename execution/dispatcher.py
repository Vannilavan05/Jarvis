"""Command Dispatcher and Security Pipeline for Jarvis.
Enforces the mandatory execution lifecycle:
Command Validation -> Authorization Check -> Confirmation -> Execution -> Verification -> Spoken Feedback -> Audit Logging.
"""

import logging
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional

try:
    from nlp.intent_parser import CommandObject
    from security.audit_logger import AuditLogger
    from security.confirmation_manager import ConfirmationManager
    from security.permissions import PermissionLevel
except (ImportError, ValueError):
    from jarvis.nlp.intent_parser import CommandObject
    from jarvis.security.audit_logger import AuditLogger
    from jarvis.security.confirmation_manager import ConfirmationManager
    from jarvis.security.permissions import PermissionLevel
from .command_registry import CommandRegistry
from .plugin_manager import PluginManager

logger = logging.getLogger(__name__)


@dataclass
class ExecutionResponse:
    success: bool
    spoken_feedback: str
    intent: Optional[str] = None
    parameters: Optional[Dict[str, Any]] = None
    data: Optional[Any] = None
    requires_confirmation: bool = False
    is_emergency: bool = False
    error: Optional[str] = None


class CommandDispatcher:
    """Central authorized command execution pipeline."""

    def __init__(
        self,
        command_registry: Optional[CommandRegistry] = None,
        plugin_manager: Optional[PluginManager] = None,
        confirmation_manager: Optional[ConfirmationManager] = None,
        audit_logger: Optional[AuditLogger] = None,
        current_permission_level: PermissionLevel = PermissionLevel.PRIVILEGED,
        context: Optional[Dict[str, Any]] = None,
    ):
        self.registry = command_registry or CommandRegistry()
        self.plugins = plugin_manager or PluginManager()
        self.confirmation_mgr = confirmation_manager or ConfirmationManager()
        self.audit = audit_logger or AuditLogger()
        self.current_permission_level = current_permission_level
        self.context = context or {}

    def dispatch(self, cmd: CommandObject) -> ExecutionResponse:
        """Executes the command through the secure pipeline."""
        raw_text = cmd.raw_text
        intent = cmd.intent
        params = cmd.parameters
        conf = cmd.confidence
        model_version = cmd.model_version or 1

        # 1. Handle Ambiguity or Clarification requests
        if cmd.is_ambiguous or not intent or intent == "unknown":
            feedback = cmd.clarification_prompt or "I am uncertain of your command. Could you please clarify?"
            self.audit.log(
                command_text=raw_text,
                intent=intent,
                confidence=conf,
                permission_level=self.current_permission_level.value,
                authorized=False,
                status="AMBIGUOUS",
                result=feedback,
                model_version=model_version,
            )
            return ExecutionResponse(
                success=False,
                spoken_feedback=feedback,
                intent=intent,
                parameters=params,
                error="Command ambiguous or low confidence",
            )

        # 2. Handle Pending Action Confirmation
        if intent == "confirm_action":
            success, feedback, exec_result = self.confirmation_mgr.verify_and_execute(raw_text)
            self.audit.log(
                command_text=raw_text,
                intent=intent,
                confidence=1.0,
                permission_level=self.current_permission_level.value,
                authorized=True,
                confirmation_required=True,
                confirmation_status="CONFIRMED" if success else "REJECTED",
                status="COMPLETED" if success else "FAILED",
                result=feedback,
                model_version=model_version,
            )
            return ExecutionResponse(
                success=success,
                spoken_feedback=feedback,
                intent=intent,
                parameters=params,
                data=exec_result,
            )

        # 3. Handle Explicit Cancellation
        if intent == "cancel_action":
            ok, msg = self.confirmation_mgr.cancel()
            feedback = msg if ok else "Action cancelled. Returning to idle."
            self.audit.log(
                command_text=raw_text,
                intent=intent,
                confidence=1.0,
                permission_level=self.current_permission_level.value,
                authorized=True,
                status="CANCELLED",
                result=feedback,
                model_version=model_version,
            )
            return ExecutionResponse(
                success=True,
                spoken_feedback=feedback,
                intent=intent,
                parameters=params,
                is_emergency=True,
            )

        # 4. Handle Emergency Stop / Listening State
        if intent == "stop_listening":
            feedback = "Deactivating listening mode. Voice commands paused."
            self.audit.log(
                command_text=raw_text,
                intent=intent,
                confidence=1.0,
                permission_level=self.current_permission_level.value,
                authorized=True,
                status="STOPPED",
                result=feedback,
                model_version=model_version,
            )
            return ExecutionResponse(
                success=True,
                spoken_feedback=feedback,
                intent=intent,
                parameters=params,
                is_emergency=True,
            )

        if intent in {"pause_listening", "resume_listening"}:
            feedback = "Listening paused." if intent == "pause_listening" else "Listening resumed."
            return ExecutionResponse(success=True, spoken_feedback=feedback, intent=intent, parameters=params)

        # LLM conversation output is display-only; it never reaches an execution plugin.
        if intent == "conversation":
            feedback = cmd.clarification_prompt or "I'm here. What would you like to do?"
            return ExecutionResponse(success=True, spoken_feedback=feedback, intent=intent, parameters={})

        # 5. Intent Registry & Authorization Validation
        meta = self.registry.get(intent)
        if not meta:
            feedback = f"Command '{intent}' is not registered or supported by Jarvis."
            self.audit.log(
                command_text=raw_text,
                intent=intent,
                confidence=conf,
                permission_level=self.current_permission_level.value,
                authorized=False,
                status="UNSUPPORTED",
                result=feedback,
                model_version=model_version,
            )
            return ExecutionResponse(
                success=False, spoken_feedback=feedback, intent=intent, error="Unsupported intent"
            )

        # Check permission rank
        if not self.current_permission_level.is_authorized(meta.permission_level):
            feedback = (
                f"Permission denied. This operation requires {meta.permission_level.value} level permissions, "
                f"but current level is {self.current_permission_level.value}."
            )
            self.audit.log(
                command_text=raw_text,
                intent=intent,
                confidence=conf,
                permission_level=meta.permission_level.value,
                authorized=False,
                status="PERMISSION_DENIED",
                result=feedback,
                model_version=model_version,
            )
            return ExecutionResponse(
                success=False, spoken_feedback=feedback, intent=intent, error="Permission denied"
            )

        # 6. Resolve Plugin Handler
        plugin = self.plugins.get_plugin_for_intent(intent)
        if not plugin:
            feedback = f"No execution plugin available for '{intent}'."
            return ExecutionResponse(
                success=False, spoken_feedback=feedback, intent=intent, error="Plugin not found"
            )

        # Capture the target window before a confirmation prompt. A later focus change
        # must not redirect a confirmed click into a different application.
        capture_target = getattr(plugin, "capture_target", None)
        if intent == "click_ui_control" and callable(capture_target):
            try:
                params = capture_target(intent, params)
            except Exception as exc:
                feedback = "I could not safely identify the target window, so I did not click anything."
                logger.warning("Could not capture desktop target before confirmation: %s", exc)
                return ExecutionResponse(success=False, spoken_feedback=feedback, intent=intent, error=str(exc))

        # 7. Check if Sensitive Action requires explicit confirmation
        if meta.requires_confirmation or meta.permission_level in {
            PermissionLevel.DESTRUCTIVE,
            PermissionLevel.PRIVILEGED,
        }:
            target_desc = f"{intent.replace('_', ' ')} with parameters {params}"
            # Define closure to be called upon user confirmation
            def action_callback():
                return plugin.execute(intent, params, self.context)

            action_id, confirm_prompt = self.confirmation_mgr.request_confirmation(
                intent=intent,
                target_description=target_desc,
                callback=action_callback,
                parameters=params,
            )

            self.audit.log(
                command_text=raw_text,
                intent=intent,
                confidence=conf,
                permission_level=meta.permission_level.value,
                authorized=True,
                confirmation_required=True,
                confirmation_status="PENDING",
                status="AWAITING_CONFIRMATION",
                result=confirm_prompt,
                model_version=model_version,
            )

            return ExecutionResponse(
                success=True,
                spoken_feedback=confirm_prompt,
                intent=intent,
                parameters=params,
                requires_confirmation=True,
            )

        # 8. Standard Execution via Plugin
        try:
            exec_result = plugin.execute(intent, params, self.context)
            self.audit.log(
                command_text=raw_text,
                intent=intent,
                confidence=conf,
                permission_level=meta.permission_level.value,
                authorized=True,
                status="COMPLETED" if exec_result.success else "FAILED",
                result=exec_result.spoken_response,
                error_details=exec_result.error,
                model_version=model_version,
            )
            return ExecutionResponse(
                success=exec_result.success,
                spoken_feedback=exec_result.spoken_response,
                intent=intent,
                parameters=params,
                data=exec_result.data,
                error=exec_result.error,
            )
        except Exception as e:
            logger.error(f"Execution error for intent {intent}: {e}")
            err_msg = f"An error occurred while executing {intent}."
            self.audit.log(
                command_text=raw_text,
                intent=intent,
                confidence=conf,
                permission_level=meta.permission_level.value,
                authorized=True,
                status="EXCEPTION",
                error_details=str(e),
                model_version=model_version,
            )
            return ExecutionResponse(
                success=False, spoken_feedback=err_msg, intent=intent, error=str(e)
            )
