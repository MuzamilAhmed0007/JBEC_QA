# main.py
import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer
from config import *
from models.bert_crf import BERT_CRF
from data.dataset_loader import QADataset
from utils.evaluation import evaluate_predictions
from query_graph.sqgg import SQGG

def train_model():
    print("Loading tokenizer and dataset...")
    tokenizer = BertTokenizer.from_pretrained(BERT_MODEL)
    dataset = QADataset(DATA_PATH)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    print("Initializing BERT-CRF model...")
    model = BERT_CRF(bert_model=BERT_MODEL, num_labels=NUM_LABELS)
    model.set_device(DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    print("Starting training...")
    for epoch in range(EPOCHS):
        print(f"Epoch {epoch + 1}")
        total_loss = 0
        for batch in dataloader:
            loss = model.training_step(batch, optimizer)
            total_loss += loss
        print(f"Loss: {total_loss:.4f}")

    model.save_model("bert_crf_model.pt")

if __name__ == "__main__":
    train_model()
