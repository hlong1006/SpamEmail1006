# Spam Email Classification System
## 📁 Project Structure

checkspamEmail/
├── data/
│   ├── raw/              # Original email CSV
│   ├── processed/        # Cleaned/processed data
│   └── split/            # Train/test split (npy files)
├── notebooks/
│   ├── 01_eda.ipynb      # Exploratory data analysis
│   ├── 02_preprocessing.ipynb
│   └── 03_modeling.ipynb
├── src/
│   ├── config.py         # Configuration & hyperparameters
│   ├── preprocessing.py  # Data cleaning & tokenization
│   ├── model.py          # Model architecture (Bidirectional LSTM)
│   ├── train.py          # Training script
│   └── predict.py        # Prediction class
├── saved_models/         # Trained model & tokenizer
│   ├── best_model.keras  # Trained neural network
│   └── tokenizer.pickle  # Text tokenizer
├── tests/
│   ├── test_preprocessing.py  # Tests for data pipeline
│   └── test_model.py          # Tests for model
├── main.py               # Main training pipeline
├── requirements.txt
└── README.md
```

## 🚀 Quick Start

### 1. Installation

```bash
pip install -r requirements.txt
```

### 2. Train Model

```bash
python main.py --data path/to/emails.csv
```

The CSV should have columns: `text`, `label` (0=HAM, 1=SPAM)


## 📊 Model Architecture

**Bidirectional LSTM with Embedding**

```
Input Layer (Max Length: 200)
    ↓
Embedding Layer (Vocab: 10000, Dim: 64)
    ↓
Bidirectional LSTM (Hidden: 64)
    ↓
Dropout (0.5)
    ↓
Dense Layer (32 units, ReLU)
    ↓
Output Layer (1 unit, Sigmoid)
    ↓
Prediction (0.0 - 1.0 probability)
```

## 📚 Main Components

### `src/config.py` - Configuration
- `VOCAB_SIZE`: 10,000
- `MAX_LEN`: 200 tokens
- `EMBEDDING_DIM`: 64
- `HIDDEN_DIM`: 64
- `EPOCHS`: 10
- `BATCH_SIZE`: 32
- `THRESHOLD`: 0.5

### `src/preprocessing.py` - Data Pipeline
- Text cleaning (URLs, emails, numbers, special chars)
- Tokenization & padding
- Train/test splitting
- Data serialization

### `src/model.py` - Neural Network
- Embedding layer
- Bidirectional LSTM
- Dropout regularization
- Binary classification output

### `src/predict.py` - Inference
- Load trained model & tokenizer
- Make predictions on new texts
- Return probability scores

### `api/app.py` - REST API
Endpoints:
- `GET /health` - Health check
- `POST /predict` - Single email prediction
- `POST /predict-batch` - Multiple emails
- `GET /model-info` - Model information



## 📈 Training Pipeline

```
main.py
├── Load CSV
├── Create Tokenizer
├── Clean & Tokenize Text
├── Pad Sequences
├── Split Train/Test
├── Save Preprocessed Data
├── Build Model
├── Train with Callbacks
│   ├── EarlyStopping
│   └── ModelCheckpoint
├── Evaluate
└── Test Predictions

## 📊 Model Performance

After training, check metrics:
- **Test Loss**: Binary crossentropy
- **Test Accuracy**: Classification accuracy
- **Spam Probability**: Sigmoid output (0-1)

## ⚙️ Dependencies

- `tensorflow>=2.10.0` - Deep learning framework
- `pandas>=1.3.0` - Data manipulation
- `numpy>=1.21.0` - Numerical computing
- `flask>=2.0.0` - Web API
- `scikit-learn>=1.0.0` - ML utilities

