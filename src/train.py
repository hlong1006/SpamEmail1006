import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from src import config
from src.model import build_model
from src.preprocessing import load_tokenizer

def train():
    print("Loading data...")
    X_train = np.load(os.path.join(config.DATA_PATH, 'X_train.npy'))
    X_test = np.load(os.path.join(config.DATA_PATH, 'X_test.npy'))
    y_train = np.load(os.path.join(config.DATA_PATH, 'y_train.npy'))
    y_test = np.load(os.path.join(config.DATA_PATH, 'y_test.npy'))

    tokenizer_path = os.path.join(config.MODEL_PATH, 'tokenizer.pickle')
    tokenizer = load_tokenizer(tokenizer_path)
    vocab_size = len(tokenizer.word_index)

    model = build_model(vocab_size=vocab_size, 
                        embedding_dim=config.EMBEDDING_DIM, 
                        max_len=config.MAX_LEN,
                        hidden_dim=config.HIDDEN_DIM)
    
    model.summary()

    checkpoint_path = os.path.join(config.MODEL_PATH)
    callbacks = [
        EarlyStopping(patience=3, monitor='val_loss', restore_best_weights=True),
        ModelCheckpoint(checkpoint_path, save_best_only=True, monitor='val_loss')
    ]

    history = model.fit(X_train, y_train,
                        epochs=config.EPOCHS,
                        batch_size=config.BATCH_SIZE,
                        validation_data=(X_test, y_test),
                        callbacks=callbacks)
    
    print(checkpoint_path)

if __name__ == "__main__":
    train()