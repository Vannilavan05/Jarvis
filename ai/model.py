"""Deep Learning Intent Classification Architecture for Jarvis.
Implemented using PyTorch with embedding, bidirectional recurrent processing, and classification layers.
"""

from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class JarvisIntentNet(nn.Module):
    """Neural Intent Classifier.

    Maps word token sequences to probability distributions over intent classes.
    """

    def __init__(
        self,
        vocab_size: int,
        num_classes: int,
        embedding_dim: int = 64,
        hidden_dim: int = 64,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.gru = nn.GRU(
            embedding_dim,
            hidden_dim,
            batch_first=True,
            bidirectional=True,
            num_layers=1,
        )
        self.fc1 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Tensor of shape [batch_size, seq_len] with token IDs.

        Returns:
            logits: Tensor of shape [batch_size, num_classes].
        """
        # [batch_size, seq_len, embedding_dim]
        embedded = self.embedding(x)
        # gru_out: [batch_size, seq_len, hidden_dim * 2]
        gru_out, _ = self.gru(embedded)
        # Global max-pooling across sequence length
        pooled = torch.max(gru_out, dim=1)[0]
        # Intermediate dense representation
        hidden = self.relu(self.fc1(pooled))
        hidden = self.dropout(hidden)
        # Logits
        logits = self.out(hidden)
        return logits

    def get_probabilities(self, x: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
        """Computes calibrated softmax probabilities for the input sequence."""
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            if temperature > 0:
                logits = logits / temperature
            probs = F.softmax(logits, dim=-1)
        return probs
