"""Sensitive Action Confirmation Manager for Jarvis.
Handles two-phase verification for destructive and privileged operations.
"""

import time
import uuid
from typing import Any, Callable, Dict, Optional, Tuple


class PendingAction:
    """Represents a sensitive operation awaiting explicit spoken confirmation."""

    def __init__(
        self,
        intent: str,
        target_description: str,
        callback: Callable[[], Any],
        parameters: Dict[str, Any],
        timeout_seconds: float = 30.0,
    ):
        self.action_id = str(uuid.uuid4())
        self.intent = intent
        self.target_description = target_description
        self.callback = callback
        self.parameters = parameters
        self.created_at = time.time()
        self.timeout_seconds = timeout_seconds

    @property
    def is_expired(self) -> bool:
        """Returns True if the confirmation window has expired."""
        return (time.time() - self.created_at) > self.timeout_seconds

    @property
    def remaining_seconds(self) -> float:
        """Returns remaining seconds before expiration, minimum 0.0."""
        rem = self.timeout_seconds - (time.time() - self.created_at)
        return max(0.0, rem)


class ConfirmationManager:
    """Manages pending confirmations for sensitive actions."""

    def __init__(self, default_timeout_seconds: float = 30.0):
        self.default_timeout_seconds = default_timeout_seconds
        self._pending: Optional[PendingAction] = None

    def request_confirmation(
        self,
        intent: str,
        target_description: str,
        callback: Callable[[], Any],
        parameters: Dict[str, Any],
        timeout_seconds: Optional[float] = None,
    ) -> Tuple[str, str]:
        """Registers a pending action and returns (action_id, prompt_message)."""
        timeout = timeout_seconds or self.default_timeout_seconds
        self._pending = PendingAction(
            intent=intent,
            target_description=target_description,
            callback=callback,
            parameters=parameters,
            timeout_seconds=timeout,
        )
        prompt = (
            f"You requested to {target_description}. "
            f"This is a sensitive operation. Please say 'confirm' or 'confirm {intent}' to authorize, "
            f"or 'cancel' to abort. You have {int(timeout)} seconds."
        )
        return self._pending.action_id, prompt

    def get_pending(self) -> Optional[PendingAction]:
        """Returns active pending action if not expired; purges if expired."""
        if self._pending is None:
            return None
        if self._pending.is_expired:
            self._pending = None
            return None
        return self._pending

    def has_pending(self) -> bool:
        """Checks if a valid, unexpired confirmation request exists."""
        return self.get_pending() is not None

    def verify_and_execute(self, confirmation_phrase: str) -> Tuple[bool, str, Any]:
        """Verifies spoken confirmation against the pending action.

        Returns (success, message, execution_result).
        """
        pending = self.get_pending()
        if pending is None:
            return False, "There are no pending actions awaiting confirmation.", None

        phrase_lower = confirmation_phrase.strip().lower()

        # Check explicit cancellation
        if any(w in phrase_lower for w in ["cancel", "abort", "no", "deny", "stop"]):
            self.cancel()
            return False, "Operation cancelled by user request.", None

        # Verify confirmation intent
        valid_confirm_tokens = ["confirm", "yes confirm", "authorize", "proceed", "yes, confirm"]
        is_confirmed = any(token in phrase_lower for token in valid_confirm_tokens)

        # Check if the confirmation mentions the intent specifically or generally
        if not is_confirmed and pending.intent.replace("_", " ") not in phrase_lower:
            return False, "Confirmation unclear. Please say 'confirm' to proceed or 'cancel' to abort.", None

        # Execute the confirmed action
        action_callback = pending.callback
        self._pending = None  # Invalidate immediately to prevent replay

        try:
            result = action_callback()
            return True, f"Action authorized and completed: {pending.target_description}", result
        except Exception as e:
            return False, f"Action failed during execution: {str(e)}", None

    def cancel(self) -> Tuple[bool, str]:
        """Cancels any pending action."""
        if self._pending is not None:
            desc = self._pending.target_description
            self._pending = None
            return True, f"Cancelled pending action: {desc}"
        return False, "No action was pending."
