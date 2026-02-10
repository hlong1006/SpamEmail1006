# Spam Email Classification System

A complete end-to-end spam detection system using deep learning (Bidirectional LSTM) with REST API and unit tests.

## 📁 Project Structure

```
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
├── api/
│   ├── app.py            # Flask REST API
│   ├── README.md         # API documentation
│   └── requirements.txt
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

### 3. Run API Server

```bash
cd api
python app.py
```

API available at `http://localhost:5000`

### 4. Make Predictions

```bash
# Single email
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "Click here to win a free iPhone!"}'

# Multiple emails
curl -X POST http://localhost:5000/predict-batch \
  -H "Content-Type: application/json" \
  -d '{"texts": ["Email 1", "Email 2", "Email 3"]}'
```

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

## 🧪 Testing

### Run Unit Tests

```bash
# Test preprocessing
python -m pytest tests/test_preprocessing.py -v

# Test model
python -m pytest tests/test_model.py -v

# Run all tests
python -m pytest tests/ -v
```

### Test Coverage

**test_preprocessing.py:**
- Text cleaning (URLs, emails, numbers, special chars)
- Tokenizer creation & vocabulary
- Padding & sequence length
- Data splitting & shuffling

**test_model.py:**
- Model creation & architecture
- Input/output shapes
- Compilation & training
- Batch predictions
- Layer verification

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
```

### Training Command

```bash
# Full pipeline (preprocess + train)
python main.py --data data/raw/emails.csv

# Skip preprocessing (use cached data)
python main.py --data data/raw/emails.csv --skip-preprocessing
```

## 🔍 Data Preprocessing

### Text Cleaning Steps:
1. Convert to lowercase
2. Remove URLs (http://, https://)
3. Remove email addresses
4. Remove numbers
5. Remove special characters & punctuation
6. Tokenize into words
7. Pad/truncate to max length (200)

Example:
```
Input:  "CLICK HERE!!! http://fake.com Prize@email.com 12345"
Output: [token_ids with padding to 200 length]
```

## 📊 Model Performance

After training, check metrics:
- **Test Loss**: Binary crossentropy
- **Test Accuracy**: Classification accuracy
- **Spam Probability**: Sigmoid output (0-1)

## 🛠️ Development

### Running Locally

```bash
# Terminal 1 - API Server
cd api
DEBUG=true python app.py

# Terminal 2 - Test requests
curl http://localhost:5000/health
```

### Configuration

Edit `src/config.py` to adjust:
- Model hyperparameters
- Data paths
- Thresholds

## 📝 API Response Examples

**Single Prediction:**
```json
{
  "text": "Click here to win!",
  "prediction": "SPAM",
  "spam_probability": 0.92,
  "is_spam": true
}
```

**Batch Prediction:**
```json
{
  "predictions": [
    {
      "text": "Congratulations!",
      "prediction": "SPAM",
      "spam_probability": 0.95,
      "is_spam": true
    },
    {
      "text": "Hi, let's talk",
      "prediction": "HAM",
      "spam_probability": 0.12,
      "is_spam": false
    }
  ],
  "total": 2
}
```

## ⚙️ Dependencies

- `tensorflow>=2.10.0` - Deep learning framework
- `pandas>=1.3.0` - Data manipulation
- `numpy>=1.21.0` - Numerical computing
- `flask>=2.0.0` - Web API
- `scikit-learn>=1.0.0` - ML utilities

## 🎯 Next Steps

- [ ] Hyperparameter tuning
- [ ] Cross-validation
- [ ] Data augmentation
- [ ] Model explainability (LIME, SHAP)
- [ ] Docker containerization
- [ ] CI/CD pipeline
- [ ] Load testing
- [ ] Model versioning

## 📄 License

MIT License
