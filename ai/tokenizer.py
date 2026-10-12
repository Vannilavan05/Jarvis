"""Text Tokenizer and Vocabulary for Jarvis Intent Classification.
Handles tokenization, vocabulary mapping, sequence padding, and persistence.
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Union
import torch


class JarvisTokenizer:
    """Vocabulary builder and text-to-tensor tokenizer for intent classification."""

    PAD_TOKEN = "<pad>"
    UNK_TOKEN = "<unk>"
    PAD_IDX = 0
    UNK_IDX = 1

    def __init__(self, max_length: int = 24):
        self.max_length = max_length
        self.word2idx: Dict[str, int] = {self.PAD_TOKEN: self.PAD_IDX, self.UNK_TOKEN: self.UNK_IDX}
        self.idx2word: Dict[int, str] = {self.PAD_IDX: self.PAD_TOKEN, self.UNK_IDX: self.UNK_TOKEN}
        self.is_fitted = False

    def clean_text(self, text: str) -> List[str]:
        """Lowercases, strips punctuation, and extracts word tokens."""
        text = text.lower().strip()
        tokens = re.findall(r"\b[a-z0-9_-]+\b", text)
        return tokens

    def fit_on_texts(self, texts: List[str]) -> "JarvisTokenizer":
        """Builds vocabulary from a list of sentence strings."""
        self.word2idx = {self.PAD_TOKEN: self.PAD_IDX, self.UNK_TOKEN: self.UNK_IDX}
        self.idx2word = {self.PAD_IDX: self.PAD_TOKEN, self.UNK_IDX: self.UNK_TOKEN}

        current_idx = 2
        for text in texts:
            tokens = self.clean_text(text)
            for token in tokens:
                if token not in self.word2idx:
                    self.word2idx[token] = current_idx
                    self.idx2word[current_idx] = token
                    current_idx += 1

        self.is_fitted = True
        return self

    def encode(self, text: str) -> List[int]:
        """Converts text into a list of vocabulary token indices."""
        tokens = self.clean_text(text)
        indices = [self.word2idx.get(t, self.UNK_IDX) for t in tokens]
        return indices

    def encode_and_pad(self, text: str) -> torch.Tensor:
        """Converts text into a fixed-length padded 1D PyTorch tensor."""
        indices = self.encode(text)
        if len(indices) < self.max_length:
            indices = indices + [self.PAD_IDX] * (self.max_length - len(indices))
        else:
            indices = indices[: self.max_length]
        return torch.tensor(indices, dtype=torch.long)

    @property
    def vocab_size(self) -> int:
        return len(self.word2idx)

    def save(self, filepath: Union[str, Path]) -> None:
        """Saves tokenizer vocabulary to JSON."""
        data = {
            "max_length": self.max_length,
            "word2idx": self.word2idx,
        }
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, filepath: Union[str, Path]) -> "JarvisTokenizer":
        """Loads tokenizer vocabulary from JSON."""
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        tokenizer = cls(max_length=data.get("max_length", 24))
        tokenizer.word2idx = data["word2idx"]
        tokenizer.idx2word = {int(v): k for k, v in tokenizer.word2idx.items()}
        tokenizer.is_fitted = True
        return tokenizer
