import mlflow
import torch
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score)
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def load_data(texts_file_path, labels_file_path):
    with open(texts_file_path, 'r', encoding='utf-8') as texts_file, \
         open(labels_file_path, 'r', encoding='utf-8') as labels_file:
        texts = texts_file.readlines()
        labels = labels_file.readlines()
        labels = [int(label.strip()) for label in labels]
        data = [{"text": text.strip(), "label": label} for text, label in zip(texts, labels)]
    return data

def evaluate_model(model, tokenizer, data):
    model.eval()
    predictions, true_labels = [], []

    for item in data:
        inputs = tokenizer(item['text'], padding=True, truncation=True, return_tensors="pt", max_length=512)
        with torch.no_grad():
            outputs = model(**inputs)
        logits = outputs.logits
        preds = torch.argmax(logits, dim=-1).numpy()
        predictions.extend(preds)
        true_labels.append(item['label'])
    
    accuracy = accuracy_score(true_labels, predictions)
    precision = precision_score(true_labels, predictions, average='weighted', zero_division=0)
    recall = recall_score(true_labels, predictions, average='weighted', zero_division=0)
    f1 = f1_score(true_labels, predictions, average='weighted', zero_division=0)
    return accuracy, precision, recall, f1

texts_file_path = '../data/test_text.txt'
labels_file_path = '../data/test_labels.txt'
data = load_data(texts_file_path, labels_file_path)

mlflow.set_experiment("experiment_distilbert-base-uncased-finetuned")

with mlflow.start_run(run_name="distilbert-base-uncased-finetuned-sst-2-english"):
    model_name = "distilbert-base-uncased-finetuned-sst-2-english"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    
    mlflow.log_param("model_name", model_name)
    
    accuracy, precision, recall, f1 = evaluate_model(model, tokenizer, data)
    
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1_score", f1)

    
