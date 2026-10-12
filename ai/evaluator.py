"""Model Evaluation Module for Jarvis.
Computes evaluation metrics, loss, per-intent accuracy, and confidence calibration.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .dataset import JarvisDataset
from .model import JarvisIntentNet


@dataclass
class EvaluationMetrics:
    total_samples: int
    accuracy: float
    average_loss: float
    average_confidence: float
    is_conclusive: bool
    summary: str
    per_intent_accuracy: Dict[str, float]


class ModelEvaluator:
    """Evaluates a trained JarvisIntentNet on test or validation splits."""

    @staticmethod
    def evaluate(
        model: JarvisIntentNet,
        dataset: Optional[JarvisDataset],
        idx_to_label: Dict[int, str],
        device: torch.device,
        criterion: Optional[nn.Module] = None,
    ) -> EvaluationMetrics:
        """Runs evaluation over the given dataset."""
        if dataset is None or len(dataset) == 0:
            return EvaluationMetrics(
                total_samples=0,
                accuracy=0.0,
                average_loss=0.0,
                average_confidence=0.0,
                is_conclusive=False,
                summary="Evaluation inconclusive: test dataset is empty or not provided.",
                per_intent_accuracy={},
            )

        if len(dataset) < 4:
            return EvaluationMetrics(
                total_samples=len(dataset),
                accuracy=0.0,
                average_loss=0.0,
                average_confidence=0.0,
                is_conclusive=False,
                summary=(
                    f"Evaluation inconclusive: dataset has only {len(dataset)} samples. "
                    "At least 4 test samples required to prevent misleading metrics."
                ),
                per_intent_accuracy={},
            )

        loss_fn = criterion or nn.CrossEntropyLoss()
        dataloader = DataLoader(dataset, batch_size=8, shuffle=False)

        model.eval()
        total_loss = 0.0
        correct = 0
        total_samples = 0
        confidences: List[float] = []

        per_intent_correct: Dict[str, int] = {}
        per_intent_total: Dict[str, int] = {}

        with torch.no_grad():
            for x_batch, y_batch in dataloader:
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)

                logits = model(x_batch)
                loss = loss_fn(logits, y_batch)
                total_loss += loss.item() * len(y_batch)

                probs = torch.softmax(logits, dim=-1)
                batch_confs, preds = torch.max(probs, dim=-1)

                correct += (preds == y_batch).sum().item()
                total_samples += len(y_batch)
                confidences.extend(batch_confs.cpu().tolist())

                for pred, actual in zip(preds.cpu().tolist(), y_batch.cpu().tolist()):
                    label_name = idx_to_label.get(actual, f"class_{actual}")
                    per_intent_total[label_name] = per_intent_total.get(label_name, 0) + 1
                    if pred == actual:
                        per_intent_correct[label_name] = per_intent_correct.get(label_name, 0) + 1

        acc = correct / total_samples if total_samples > 0 else 0.0
        avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
        avg_conf = sum(confidences) / len(confidences) if confidences else 0.0

        per_intent_acc = {}
        for intent, tot in per_intent_total.items():
            per_intent_acc[intent] = per_intent_correct.get(intent, 0) / tot if tot > 0 else 0.0

        summary = (
            f"Evaluated {total_samples} samples. "
            f"Accuracy: {acc * 100:.1f}%, Average Loss: {avg_loss:.4f}, "
            f"Mean Confidence: {avg_conf * 100:.1f}%."
        )

        return EvaluationMetrics(
            total_samples=total_samples,
            accuracy=acc,
            average_loss=avg_loss,
            average_confidence=avg_conf,
            is_conclusive=True,
            summary=summary,
            per_intent_accuracy=per_intent_acc,
        )
