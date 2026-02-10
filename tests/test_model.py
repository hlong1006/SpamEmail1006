"""Tests cho kiến trúc mô hình và build."""
import numpy as np
import pytest

from src.model import build_model
from src import config


def test_build_model_shape():
    vocab_size = 1000
    model = build_model(
        vocab_size=vocab_size,
        embedding_dim=config.EMBEDDING_DIM,
        max_len=config.MAX_LEN,
        hidden_dim=config.HIDDEN_DIM,
    )
    assert model.input_shape == (None, config.MAX_LEN)
    assert model.output_shape == (None, 1)


def test_model_forward_pass():
    model = build_model(
        vocab_size=500,
        embedding_dim=16,
        max_len=20,
        hidden_dim=16,
    )
    X = np.random.randint(0, 500, size=(4, 20))
    y = model.predict(X, verbose=0)
    assert y.shape == (4, 1)
    assert np.all((y >= 0) & (y <= 1))


def test_model_compile():
    model = build_model(
        vocab_size=100,
        embedding_dim=config.EMBEDDING_DIM,
        max_len=config.MAX_LEN,
        hidden_dim=config.HIDDEN_DIM,
    )
    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
    X = np.random.randint(0, 100, size=(8, config.MAX_LEN))
    y = np.random.randint(0, 2, size=(8, 1))
    history = model.fit(X, y, epochs=1, verbose=0)
    assert "loss" in history.history
