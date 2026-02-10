"""Tests cho tiền xử lý văn bản và tokenizer."""
import numpy as np
import pandas as pd
import pytest

from src.preprocessing import (
    clean_text,
    create_tokenizer,
    tokenize_and_pad,
    split_data,
    preprocess_data,
    load_tokenizer,
    save_tokenizer,
    preprocess_texts,
)
from src import config


def test_clean_text_lowercase():
    assert clean_text("HELLO World") == "hello world"


def test_clean_text_removes_url():
    t = clean_text("Visit http://example.com/page for more")
    assert "http" not in t
    assert "visit" in t and "for" in t and "more" in t


def test_clean_text_removes_email():
    t = clean_text("Contact me at user@domain.com please")
    assert "@" not in t
    assert "contact" in t and "please" in t


def test_clean_text_removes_numbers():
    assert "123" not in clean_text("Price is 123 dollars")
    assert "456" not in clean_text("456")


def test_clean_text_removes_special_chars():
    t = clean_text("Hello!!! What???")
    assert "!!!" not in t and "???" not in t


def test_create_tokenizer_vocab():
    texts = ["hello world", "world of python", "hello python"]
    tok = create_tokenizer(texts, vocab_size=100)
    assert len(tok.word_index) >= 1
    seq = tok.texts_to_sequences(["hello world"])
    assert len(seq[0]) == 2


def test_tokenize_and_pad_length():
    texts = ["one two three", "a b"]
    tok = create_tokenizer(texts, vocab_size=100)
    padded = tokenize_and_pad(texts, tok, max_len=5)
    assert padded.shape == (2, 5)


def test_split_data_ratio():
    X = np.arange(100).reshape(50, 2)
    y = np.zeros(50)
    X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.2, random_state=42)
    assert len(X_train) == 40 and len(X_test) == 10
    assert len(y_train) == 40 and len(y_test) == 10


def test_preprocess_data_dataframe():
    df = pd.DataFrame({"text": ["hello world", "spam here"], "label": [0, 1]})
    tok = create_tokenizer(df["text"].tolist(), vocab_size=100)
    X, y = preprocess_data(df, tok, max_len=10)
    assert X.shape[0] == 2 and X.shape[1] == 10
    assert list(y) == [0, 1]


def test_preprocess_texts():
    tok = create_tokenizer(["hello world"], vocab_size=100)
    X = preprocess_texts(["hello world", "unknown words"], tok, max_len=5)
    assert X.shape == (2, 5)


def test_save_load_tokenizer(tmp_path):
    tok = create_tokenizer(["a b c"], vocab_size=100)
    path = tmp_path / "tok.pickle"
    save_tokenizer(tok, str(path))
    loaded = load_tokenizer(str(path))
    assert loaded.word_index == tok.word_index
