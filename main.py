import argparse
import os
import sys

import numpy as np
import pandas as pd

from src import config
from src.preprocessing import (
    create_tokenizer,
    preprocess_data,
    split_data,
    save_tokenizer,
)
from src.train import train
def normalize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    initial_rows = len(df)
    df = df.dropna(subset=["text", "label"])
    removed_rows = initial_rows - len(df)
    df["label"] = df["label"].astype(int)
    df["text"] = df["text"].astype(str) 
    return df


def run_preprocessing(data_path: str) -> None:
    print(f" {data_path}")
    df = pd.read_csv(data_path)
    df = normalize_dataframe(df)
    tokenizer = create_tokenizer(
        df["text"].astype(str).tolist(),
        vocab_size=config.VOCAB_SIZE,
    )

    os.makedirs(config.MODEL_PATH, exist_ok=True)
    tokenizer_path = os.path.join(config.MODEL_PATH, "tokenizer.pickle")
    save_tokenizer(tokenizer, tokenizer_path)
    X, y = preprocess_data(df, tokenizer, config.MAX_LEN)
    X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.2, random_state=42)

    os.makedirs(config.DATA_PATH, exist_ok=True)
    np.save(os.path.join(config.DATA_PATH, "X_train.npy"), X_train)
    np.save(os.path.join(config.DATA_PATH, "X_test.npy"), X_test)
    np.save(os.path.join(config.DATA_PATH, "y_train.npy"), y_train)
    np.save(os.path.join(config.DATA_PATH, "y_test.npy"), y_test)


def main():
    parser = argparse.ArgumentParser(description="Train mô hình phân loại spam email")
    parser.add_argument(
        "--data",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "data", "raw", "spam_or_not_spam.csv"),
        help="Đường dẫn file CSV (cột text/email và label)",
    )
    parser.add_argument(
        "--skip-preprocessing",
        action="store_true",
        help="data/split và tokenizer ",
    )
    args = parser.parse_args()

    if not args.skip_preprocessing:
        if not os.path.isfile(args.data):
            print(f"Lỗi: không tìm thấy file {args.data}")
            sys.exit(1)
        run_preprocessing(args.data)
    else:
        print("")

    train()


if __name__ == "__main__":
    main()
