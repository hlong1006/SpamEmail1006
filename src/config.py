import os 
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data" , "processed")
MODEL_PATH = os.path.join(BASE_DIR,"saved_models")
VOCAB_SIZE = 10000
MAX_LEN = 200 

EMBEDDING_SIM = 64 # mỗi từ được biểu diễn bằng vecto 64 chiều 
HIDDEN_DIM = 64 # Số lượng nouron trong lớp LSTM 
EPOCHS = 10 # số vòng lặp huấn luyện tối đa 
BATCH_SIZE = 32 # số lượng mẫu dữ liệu nạp vào mỗi lần cập nhật
THRESHOLD = 0.5 # ngưỡng phân loại spam
LEARNING_RATE = 0.001