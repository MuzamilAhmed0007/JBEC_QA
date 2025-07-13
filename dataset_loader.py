# data/dataset_loader.py
import json
import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer
import os

class QADataset(Dataset):
    def __init__(self, file_path, tokenizer_name="bert-base-uncased", max_len=128):
        self.tokenizer = BertTokenizer.from_pretrained(tokenizer_name)
        self.max_len = max_len
        self.data = self.load_data(file_path)

    def load_data(self, file_path):
        data = []
        ext = os.path.splitext(file_path)[1]
        
        with open(file_path, 'r', encoding='utf-8') as f:
            if ext == ".json":
                records = json.load(f)
            elif ext == ".jsonl":
                records = [json.loads(line.strip()) for line in f]
            else:
                raise ValueError("Unsupported file format")

            for record in records:
                question = record.get("question", "")
                entities = record.get("entities", [])
                relations = record.get("relations", [])
                label_ids = self.encode_labels(question, entities, relations)
                data.append((question, label_ids))
        return data

    def encode_labels(self, question, entities, relations):
        # Dummy BIO-style encoding for entities; you should enhance this based on CRF tagset
        tokens = self.tokenizer.tokenize(question)
        labels = [0] * len(tokens)

        for entity in entities:
            if entity.lower() in question.lower():
                idx = question.lower().split().index(entity.lower())
                labels[idx] = 1  # B-ENT
                if idx + 1 < len(tokens):
                    labels[idx + 1] = 2  # I-ENT

        return labels[:self.max_len] + [0] * (self.max_len - len(labels))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        question, label_ids = self.data[idx]
        encoding = self.tokenizer(
            question,
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt"
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "labels": torch.tensor(label_ids, dtype=torch.long)
        }
