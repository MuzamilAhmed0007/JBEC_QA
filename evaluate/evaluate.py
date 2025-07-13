# evaluate.py
import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer
from config import *
from models.bert_crf import BERT_CRF
from data.dataset_loader import QADataset
from utils.evaluation import evaluate_predictions

def evaluate_model():
    print("Loading tokenizer and dataset...")
    tokenizer = BertTokenizer.from_pretrained(BERT_MODEL)
    dataset = QADataset(DATA_PATH)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE)

    print("Loading saved model...")
    model = BERT_CRF(bert_model=BERT_MODEL, num_labels=NUM_LABELS)
    model.set_device(DEVICE)
    model.load_model("bert_crf_model.pt")

    print("Evaluating...")
    all_preds = []
    all_labels = []

    for batch in dataloader:
        preds, labels = model.evaluate_step(batch)
        for p, l in zip(preds, labels):
            all_preds.extend(p)
            all_labels.extend(l)

    evaluate_predictions(all_labels, all_preds)

if __name__ == "__main__":
    evaluate_model()
