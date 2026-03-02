import os
import numpy as np
import tensorflow as tf
from src import config
from src.preprocessing import load_tokenizer, preprocess_texts

class SpamClassifier:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self._load_artifacts()

    def _load_artifacts(self):
        try:
            model_path = os.path.join(config.MODEL_PATH, "best_model.keras")
            tokenizer_path = os.path.join(config.MODEL_PATH, "tokenizer.pickle")
            self.model = tf.keras.models.load_model(model_path)
            self.tokenizer = load_tokenizer(tokenizer_path)
        except Exception as e:
            print(f"Error loading artifacts: {e}")
            raise e

    def predict(self, text):
        if not text:
            return {"error": "Empty text"}

        processed_text = preprocess_texts([text], self.tokenizer, config.MAX_LEN)

        prob = self.model.predict(processed_text, verbose=0)[0][0]
        prob = np.nan_to_num(prob, nan=0.5)

        is_spam = prob > config.THRESHOLD
        label = "SPAM" if is_spam else "HAM"

        return {
            "text": text,
            "prediction": label,
            "spam_probability": float(prob),
            "is_spam": bool(is_spam)
        }

