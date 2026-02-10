# -*- coding: utf-8 -*-
"""
Flask REST API cho mô hình phân loại spam email.
Chạy từ thư mục gốc dự án: python api/app.py hoặc python -m api.app
"""
import os
import sys

# Cho phép import src khi chạy từ thư mục api/
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from flask import Flask, request, jsonify

from src.predict import SpamClassifier
from src import config

app = Flask(__name__)
classifier = None


def get_classifier():
    global classifier
    if classifier is None:
        classifier = SpamClassifier()
    return classifier


@app.route("/health", methods=["GET"])
def health():
    """Kiểm tra API hoạt động."""
    return jsonify({"status": "ok", "service": "checkspamemail"})


@app.route("/model-info", methods=["GET"])
def model_info():
    """Thông tin cấu hình mô hình."""
    return jsonify({
        "max_len": config.MAX_LEN,
        "vocab_size": config.VOCAB_SIZE,
        "threshold": config.THRESHOLD,
        "embedding_dim": config.EMBEDDING_DIM,
        "hidden_dim": config.HIDDEN_DIM,
    })


@app.route("/predict", methods=["POST"])
def predict():
    """Dự đoán một email (spam hay ham). Body JSON: {"text": "nội dung email"}."""
    data = request.get_json(force=True, silent=True) or {}
    text = data.get("text", "").strip()
    if not text:
        return jsonify({"error": "Thiếu trường 'text' trong body JSON"}), 400
    try:
        result = get_classifier().predict(text)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/predict-batch", methods=["POST"])
def predict_batch():
    """Dự đoán nhiều email. Body JSON: {"texts": ["email1", "email2", ...]}."""
    data = request.get_json(force=True, silent=True) or {}
    texts = data.get("texts", [])
    if not isinstance(texts, list):
        return jsonify({"error": "'texts' phải là mảng chuỗi"}), 400
    try:
        clf = get_classifier()
        predictions = [clf.predict(t if isinstance(t, str) else str(t)) for t in texts]
        return jsonify({"predictions": predictions, "total": len(predictions)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    import os as _os
    debug = _os.environ.get("DEBUG", "").lower() in ("1", "true", "yes")
    port = int(_os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=debug)
