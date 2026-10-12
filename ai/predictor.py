"""Intent Predictor for Jarvis.
Performs PyTorch inference to classify spoken commands into intents with confidence scores.
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import torch

from .model import JarvisIntentNet
from .model_registry import ModelRegistry
from .tokenizer import JarvisTokenizer

logger = logging.getLogger(__name__)


@dataclass
class PredictionResult:
    intent: Optional[str]
    confidence: float
    is_confident: bool
    top_candidates: List[Tuple[str, float]]
    model_version: Optional[int]


class IntentPredictor:
    """Performs inference using the active PyTorch model checkpoint."""

    def __init__(
        self,
        registry: Optional[ModelRegistry] = None,
        confidence_threshold: float = 0.65,
        device: Optional[torch.device] = None,
    ):
        self.registry = registry or ModelRegistry()
        self.confidence_threshold = confidence_threshold
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model: Optional[JarvisIntentNet] = None
        self.tokenizer: Optional[JarvisTokenizer] = None
        self.idx_to_label: Dict[int, str] = {}
        self.active_version: Optional[int] = None
        self.metadata: Dict[str, Any] = {}

        self.reload()

    def reload(self) -> bool:
        """Loads or reloads the active checkpoint from registry."""
        active_ver = self.registry.get_active_version()
        if not active_ver:
            logger.info("No active model checkpoint found in registry.")
            return False

        loaded = self.registry.load_version(active_ver, self.device)
        if not loaded:
            return False

        self.model, self.tokenizer, self.idx_to_label, self.metadata = loaded
        self.active_version = active_ver
        logger.info(f"Loaded active model version {active_ver} ({len(self.idx_to_label)} intent classes).")
        return True

    def predict(self, text: str) -> PredictionResult:
        """Runs forward inference on text and returns PredictionResult."""
        if self.model is None or self.tokenizer is None or not self.idx_to_label:
            return PredictionResult(
                intent=None,
                confidence=0.0,
                is_confident=False,
                top_candidates=[],
                model_version=None,
            )

        self.model.eval()
        with torch.no_grad():
            x = self.tokenizer.encode_and_pad(text).unsqueeze(0).to(self.device)
            probs = self.model.get_probabilities(x).squeeze(0)

            top_k = min(3, len(self.idx_to_label))
            values, indices = torch.topk(probs, k=top_k)

            top_candidates = []
            for val, idx in zip(values.cpu().tolist(), indices.cpu().tolist()):
                top_candidates.append((self.idx_to_label.get(idx, f"unknown_{idx}"), float(val)))

            best_intent, best_conf = top_candidates[0]
            is_confident = best_conf >= self.confidence_threshold

            return PredictionResult(
                intent=best_intent,
                confidence=best_conf,
                is_confident=is_confident,
                top_candidates=top_candidates,
                model_version=self.active_version,
            )
