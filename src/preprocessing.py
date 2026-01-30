import re 
import numpy as np 
import pandas as pd 
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+', '', text)   
    text = re.sub(r'\S+@\S+', '', text)   
    text = re.sub(r'\d+', '', text)       
    text = re.sub(r'[^\w\s]', '', text)
    return text

def preprocess_data(df, tokenizer, max_len):
    df['cleaned_text'] = df['text'].apply(clean_text)
    sequences = tokenizer.texts_to_sequences(df['cleaned_text'])
    padded_sequences = pad_sequences(sequences, maxlen=max_len, padding='post', truncating='post')
    labels = df['label'].values
    return padded_sequences, labels

def load_data(file_path, tokenizer, max_len):
    df = pd.read_csv(file_path)
    X, y = preprocess_data(df, tokenizer, max_len)
    return X, y

def save_preprocessed_data(X, y, X_path, y_path):
    np.save(X_path, X)
    np.save(y_path, y)

def load_preprocessed_data(X_path, y_path):
    X = np.load(X_path)
    y = np.load(y_path)
    return X, y

def split_data(X, y, test_size=0.2, random_state=42):
    np.random.seed(random_state)
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    X = X[indices]
    y = y[indices]
    split_index = int(X.shape[0] * (1 - test_size))
    X_train, X_test = X[:split_index], X[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]
    return X_train, X_test, y_train, y_test

def create_tokenizer(texts, vocab_size):
    tokenizer = Tokenizer(num_words=vocab_size, oov_token='<OOV>')
    tokenizer.fit_on_texts(texts)
    return tokenizer

def tokenize_and_pad(texts, tokenizer, max_len):
    sequences = tokenizer.texts_to_sequences(texts)
    padded_sequences = pad_sequences(sequences, maxlen=max_len, padding='post', truncating='post')
    return padded_sequences

def preprocess_and_split(file_path, tokenizer, max_len, test_size=0.2, random_state=42):
    df = pd.read_csv(file_path)
    X, y = preprocess_data(df, tokenizer, max_len)
    X_train, X_test, y_train, y_test = split_data(X, y, test_size, random_state)
    return X_train, X_test, y_train, y_test

def preprocess_texts(texts, tokenizer, max_len):
    cleaned_texts = [clean_text(text) for text in texts]
    padded_sequences = tokenize_and_pad(cleaned_texts, tokenizer, max_len)
    return padded_sequences

def save_tokenizer(tokenizer, file_path):
    with open(file_path, 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
def load_tokenizer(file_path):
    with open(file_path, 'rb') as handle:
        tokenizer = pickle.load(handle)
    return tokenizer
      