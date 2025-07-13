# inference.py
import torch
from transformers import BertTokenizer
from config import *
from models.bert_crf import BERT_CRF
from query_graph.sqgg import SQGG
from features.syntax_features import SyntaxFeatures
from utils.helpers import extract_entities_from_labels

def run_inference(question):
    print(f"Question: {question}")
    tokenizer = BertTokenizer.from_pretrained(BERT_MODEL)
    inputs = tokenizer(question, return_tensors="pt", truncation=True, padding=True, max_length=MAX_SEQ_LENGTH)

    model = BERT_CRF(bert_model=BERT_MODEL, num_labels=NUM_LABELS)
    model.set_device(DEVICE)
    model.load_model("bert_crf_model.pt")

    input_ids = inputs["input_ids"].to(DEVICE)
    attention_mask = inputs["attention_mask"].to(DEVICE)
    with torch.no_grad():
        predictions = model(input_ids, attention_mask)

    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
    labels = predictions[0]
    print("Predicted labels:", labels)

    extracted_entities = extract_entities_from_labels(tokens, labels)
    print("Extracted Entities:", extracted_entities)

    sqgg = SQGG()
    syntaxer = SyntaxFeatures()
    tree = syntaxer.extract_pos_tree(question)

    print("Syntax Tree:")
    syntaxer.visualize_tree(tree)

    candidate_entities = ["Eiffel_Tower", "Paris", "France"]  # Simulated candidates
    query_graph = sqgg.generate_graph(question, candidate_entities, extracted_entities)
    print("\nGenerated Query Graph:")
    for triple in query_graph:
        print(triple)

if __name__ == "__main__":
    run_inference("Where is the Eiffel Tower located?")
