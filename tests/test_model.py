"""Tests for PyTorch Deep Learning Model Architecture, Tokenizer, and Inference.
"""

import tempfile
from pathlib import Path
import pytest
import torch

from ai.model import JarvisIntentNet
from ai.tokenizer import JarvisTokenizer


def test_tokenizer_encoding_and_padding():
    tokenizer = JarvisTokenizer(max_length=6)
    texts = ["open chrome browser", "read file test.txt"]
    tokenizer.fit_on_texts(texts)

    assert tokenizer.is_fitted
    assert tokenizer.vocab_size >= 6

    # Test encode and pad
    t = tokenizer.encode_and_pad("open chrome")
    assert isinstance(t, torch.Tensor)
    assert t.shape == (6,)
    # First two should be token indices, rest should be padding (0)
    assert t[0].item() != 0
    assert t[1].item() != 0
    assert t[2].item() == 0

    # Test unknown token handling
    t_unk = tokenizer.encode_and_pad("unseen_alien_word")
    assert t_unk[0].item() == JarvisTokenizer.UNK_IDX


def test_tokenizer_save_load():
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "vocab.json"
        tok1 = JarvisTokenizer(max_length=8)
        tok1.fit_on_texts(["launch browser", "system status"])
        tok1.save(path)

        tok2 = JarvisTokenizer.load(path)
        assert tok2.is_fitted
        assert tok2.vocab_size == tok1.vocab_size
        assert tok2.encode("launch") == tok1.encode("launch")


def test_pytorch_intent_model_forward():
    vocab_size = 20
    num_classes = 4
    model = JarvisIntentNet(vocab_size=vocab_size, num_classes=num_classes, embedding_dim=16, hidden_dim=16)

    # Input batch of 3 samples, length 10
    x = torch.randint(0, vocab_size, (3, 10))
    logits = model(x)
    assert logits.shape == (3, num_classes)

    probs = model.get_probabilities(x)
    assert probs.shape == (3, num_classes)
    # Probabilities should sum to 1.0 along class dimension
    sums = probs.sum(dim=-1)
    for s in sums:
        assert abs(s.item() - 1.0) < 1e-4
