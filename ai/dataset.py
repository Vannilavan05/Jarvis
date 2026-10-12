"""Dataset management and PyTorch Dataset creation for Jarvis.
Handles data splitting, label encoding, and dataset validation.
"""

import logging
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import torch
from torch.utils.data import Dataset

from .tokenizer import JarvisTokenizer

logger = logging.getLogger(__name__)


class JarvisDataset(Dataset):
    """PyTorch Dataset yielding (input_tensor, label_tensor)."""

    def __init__(self, texts: List[str], label_indices: List[int], tokenizer: JarvisTokenizer):
        self.texts = texts
        self.label_indices = label_indices
        self.tokenizer = tokenizer

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        text = self.texts[idx]
        x = self.tokenizer.encode_and_pad(text)
        y = torch.tensor(self.label_indices[idx], dtype=torch.long)
        return x, y


@dataclass
class DatasetSplits:
    train_dataset: JarvisDataset
    val_dataset: Optional[JarvisDataset]
    test_dataset: Optional[JarvisDataset]
    label_to_idx: Dict[str, int]
    idx_to_label: Dict[int, str]
    is_sufficient: bool
    status_message: str


class DatasetManager:
    """Validates raw database examples, encodes labels, and creates splits."""

    MINIMUM_SAMPLES_FOR_EVAL = 8

    @classmethod
    def prepare_datasets(
        cls,
        examples: List[Dict[str, Any]],
        tokenizer: JarvisTokenizer,
        seed: int = 42,
    ) -> DatasetSplits:
        """Validates examples and splits them into training, validation, and test datasets."""
        # 1. Validation & Filtering
        valid_records: List[Tuple[str, str]] = []
        for ex in examples:
            text = ex.get("text", "").strip()
            intent = ex.get("intent", "").strip()
            approved = ex.get("approved", True)
            if not text or not intent or not approved:
                continue
            valid_records.append((text, intent))

        total_samples = len(valid_records)
        if total_samples < 4:
            return DatasetSplits(
                train_dataset=JarvisDataset([], [], tokenizer),
                val_dataset=None,
                test_dataset=None,
                label_to_idx={},
                idx_to_label={},
                is_sufficient=False,
                status_message=f"Dataset too small ({total_samples} examples). At least 4 required to train.",
            )

        # 2. Extract unique labels and fit tokenizer
        all_texts = [r[0] for r in valid_records]
        unique_intents = sorted(list({r[1] for r in valid_records}))
        label_to_idx = {intent: i for i, intent in enumerate(unique_intents)}
        idx_to_label = {i: intent for intent, i in label_to_idx.items()}

        if not tokenizer.is_fitted:
            tokenizer.fit_on_texts(all_texts)

        # 3. Shuffle with fixed seed for reproducibility
        rng = random.Random(seed)
        shuffled = valid_records.copy()
        rng.shuffle(shuffled)

        # 4. Partition
        if total_samples < cls.MINIMUM_SAMPLES_FOR_EVAL:
            # Small dataset: train on all, warn that evaluation is inconclusive
            train_texts = [r[0] for r in shuffled]
            train_labels = [label_to_idx[r[1]] for r in shuffled]
            return DatasetSplits(
                train_dataset=JarvisDataset(train_texts, train_labels, tokenizer),
                val_dataset=None,
                test_dataset=None,
                label_to_idx=label_to_idx,
                idx_to_label=idx_to_label,
                is_sufficient=True,
                status_message=(
                    f"Trained on {total_samples} examples. Dataset is below {cls.MINIMUM_SAMPLES_FOR_EVAL} samples, "
                    "so split evaluation was omitted to prevent misleading metrics."
                ),
            )

        # Stratified or percentage split
        n_test = max(1, int(total_samples * 0.15))
        n_val = max(1, int(total_samples * 0.15))
        n_train = total_samples - n_test - n_val

        test_records = shuffled[:n_test]
        val_records = shuffled[n_test : n_test + n_val]
        train_records = shuffled[n_test + n_val :]

        train_ds = JarvisDataset(
            [r[0] for r in train_records],
            [label_to_idx[r[1]] for r in train_records],
            tokenizer,
        )
        val_ds = JarvisDataset(
            [r[0] for r in val_records],
            [label_to_idx[r[1]] for r in val_records],
            tokenizer,
        )
        test_ds = JarvisDataset(
            [r[0] for r in test_records],
            [label_to_idx[r[1]] for r in test_records],
            tokenizer,
        )

        return DatasetSplits(
            train_dataset=train_ds,
            val_dataset=val_ds,
            test_dataset=test_ds,
            label_to_idx=label_to_idx,
            idx_to_label=idx_to_label,
            is_sufficient=True,
            status_message=f"Prepared {n_train} train, {n_val} val, {n_test} test samples across {len(unique_intents)} intents.",
        )
