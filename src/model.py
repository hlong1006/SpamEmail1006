from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional

def build_model(vocab_size, embedding_dim, max_len, hidden_dim):
    model = Sequential()
    model.add(Embedding(input_dim=vocab_size + 1, 
                        output_dim=embedding_dim, 
                        input_length=max_len))
    
    model.add(Bidirectional(LSTM(hidden_dim, return_sequences=False)))
    model.add(Dropout(0.5))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', 
                  optimizer='adam', 
                  metrics=['accuracy'])
    return model