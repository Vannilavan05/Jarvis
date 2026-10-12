"""Confidence Management and Clarification Generator for Jarvis.
"""

from typing import List, Optional, Tuple


class ConfidenceManager:
    """Evaluates prediction certainty and generates clarification prompts."""

    def __init__(self, threshold: float = 0.65):
        self.threshold = threshold

    def evaluate_confidence(
        self,
        intent: Optional[str],
        confidence: float,
        top_candidates: List[Tuple[str, float]],
    ) -> Tuple[bool, Optional[str]]:
        """Evaluates whether prediction is confident enough for execution.

        Returns (is_confident, clarification_prompt).
        """
        if not intent:
            return False, "I could not recognize that command. Could you please rephrase?"

        if confidence < self.threshold:
            if len(top_candidates) >= 2:
                cand1 = top_candidates[0][0].replace("_", " ")
                cand2 = top_candidates[1][0].replace("_", " ")
                return (
                    False,
                    f"I am not entirely certain. Did you want to {cand1} or {cand2}?",
                )
            return (
                False,
                f"Confidence is low ({confidence * 100:.0f}%). Could you please repeat the command clearly?",
            )

        return True, None
