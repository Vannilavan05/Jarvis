"""Model Training Pipeline for Jarvis.
Trains the PyTorch neural intent classifier with progress updates and evaluation.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .dataset import DatasetManager
from .evaluator import EvaluationMetrics, ModelEvaluator
from .model import JarvisIntentNet
from .model_registry import ModelRegistry
from .tokenizer import JarvisTokenizer

logger = logging.getLogger(__name__)


@dataclass
class TrainingResult:
    success: bool
    version: int
    epochs_trained: int
    final_loss: float
    train_accuracy: float
    evaluation: EvaluationMetrics
    summary_message: str
    model_path: Optional[str] = None
    vocab_path: Optional[str] = None


class ModelTrainer:
    """Orchestrates PyTorch training loops for JarvisIntentNet."""

    def __init__(
        self,
        registry: Optional[ModelRegistry] = None,
        device: Optional[torch.device] = None,
    ):
        self.registry = registry or ModelRegistry()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def train(
        self,
        examples: List[Dict[str, Any]],
        epochs: int = 35,
        lr: float = 0.005,
        batch_size: int = 8,
        progress_callback: Optional[Callable[[int, int, float, float], None]] = None,
    ) -> TrainingResult:
        """Runs the training and evaluation workflow."""
        tokenizer = JarvisTokenizer()
        splits = DatasetManager.prepare_datasets(examples, tokenizer)

        if not splits.is_sufficient or len(splits.train_dataset) == 0:
            return TrainingResult(
                success=False,
                version=self.registry.get_latest_version(),
                epochs_trained=0,
                final_loss=0.0,
                train_accuracy=0.0,
                evaluation=EvaluationMetrics(0, 0.0, 0.0, 0.0, False, splits.status_message, {}),
                summary_message=splits.status_message,
            )

        num_classes = len(splits.label_to_idx)
        model = JarvisIntentNet(
            vocab_size=tokenizer.vocab_size,
            num_classes=num_classes,
            embedding_dim=64,
            hidden_dim=64,
            dropout=0.25,
        ).to(self.device)

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        train_loader = DataLoader(splits.train_dataset, batch_size=batch_size, shuffle=True)

        last_loss = 0.0
        last_train_acc = 0.0

        for epoch in range(1, epochs + 1):
            model.train()
            running_loss = 0.0
            correct = 0
            total = 0

            for x_batch, y_batch in train_loader:
                x_batch = x_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                optimizer.zero_grad()
                logits = model(x_batch)
                loss = criterion(logits, y_batch)
                loss.backward()
                optimizer.step()

                running_loss += loss.item() * len(y_batch)
                preds = torch.argmax(logits, dim=-1)
                correct += (preds == y_batch).sum().item()
                total += len(y_batch)

            last_loss = running_loss / total if total > 0 else 0.0
            last_train_acc = correct / total if total > 0 else 0.0

            if progress_callback:
                progress_callback(epoch, epochs, last_loss, last_train_acc)

        # Evaluate model on test split (or validation split)
        eval_dataset = splits.test_dataset or splits.val_dataset
        eval_metrics = ModelEvaluator.evaluate(
            model=model,
            dataset=eval_dataset,
            idx_to_label=splits.idx_to_label,
            device=self.device,
            criterion=criterion,
        )

        # Save checkpoint in registry
        new_version, model_path, vocab_path = self.registry.save_checkpoint(
            model=model,
            tokenizer=tokenizer,
            label_to_idx=splits.label_to_idx,
            idx_to_label=splits.idx_to_label,
            dataset_size=len(examples),
            accuracy=eval_metrics.accuracy if eval_metrics.is_conclusive else last_train_acc,
            loss=last_loss,
        )

        summary = (
            f"Training complete. Model version {new_version} created. "
            f"Trained for {epochs} epochs with final loss {last_loss:.4f} and "
            f"accuracy {last_train_acc * 100:.1f}%. "
            f"{eval_metrics.summary}"
        )

        return TrainingResult(
            success=True,
            version=new_version,
            epochs_trained=epochs,
            final_loss=last_loss,
            train_accuracy=last_train_acc,
            evaluation=eval_metrics,
            summary_message=summary,
            model_path=str(model_path),
            vocab_path=str(vocab_path),
        )
