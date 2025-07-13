BERT_MODEL = "bert-base-uncased"
MAX_SEQ_LENGTH = 128
BATCH_SIZE = 16
EPOCHS = 3
LEARNING_RATE = 3e-5
NUM_LABELS = 10  # Adjust based on your dataset
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATA_PATH = "data/lcquad2.json"  # or your preferred dataset
