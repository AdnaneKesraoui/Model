#!/usr/bin/env python
# coding: utf-8

# In[31]:


import csv
import urllib.request

import great_expectations as ge
import mlflow
import numpy as np
import pandas as pd
from scipy.special import softmax
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score)
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from zenml import pipeline, step

label_mapping = {'negative': 0, 'neutral': 1, 'positive': 2}


@step
def read_tweets_from_file(file_path: str) -> list:
    with open(file_path, 'r', encoding='utf-8') as file:
        tweets = file.readlines()
    return [tweet.strip() for tweet in tweets]

@step
def read_labels_from_file(labels_path: str) -> list:
    with open(labels_path, 'r', encoding='utf-8') as file:
        labels = [int(line.strip()) for line in file]
    return labels

@step
def preprocess_step(texts: list) -> list:
    preprocessed_texts = []
    for text in texts:
        new_text = []
        for t in text.split(" "):
            t = '@user' if t.startswith('@') and len(t) > 1 else t
            t = 'http' if t.startswith('http') else t
            new_text.append(t)
        preprocessed_texts.append(" ".join(new_text))
    return preprocessed_texts


# In[32]:


import uuid

from cassandra.cluster import Cluster


@step
def insert_preprocessed_tweets_into_cassandra(processed_texts: list):
  

    CASSANDRA_CLUSTER = ['localhost']
    KEYSPACE = 'mykeyspace'
    TABLE_NAME = 'preprocessed_tweets'

    cluster = Cluster(CASSANDRA_CLUSTER)
    session = cluster.connect(KEYSPACE)

    def insert_preprocessed_tweet(tweet_text):
        query = f"INSERT INTO {TABLE_NAME} (id, tweet_text) VALUES (%s, %s)"
        session.execute(query, (uuid.uuid4(), tweet_text))

    for tweet_text in processed_texts:
        stored_output=insert_preprocessed_tweet(tweet_text)
    
    print("All preprocessed tweets have been inserted into Cassandra.")
    


# In[33]:


# from transformers import AutoModelForSequenceClassification, AutoTokenizer
# import torch
# from torch.utils.data import Dataset, DataLoader
# from torch.optim import AdamW
# from tqdm import tqdm

# class TweetsDataset(Dataset):
#     def __init__(self, texts, labels, tokenizer, max_length=512):
#         self.tokenizer = tokenizer
#         self.texts = texts
#         self.labels = labels
#         self.max_length = max_length

#     def __len__(self):
#         return len(self.texts)

#     def __getitem__(self, idx):
#         text = self.texts[idx]
#         labels = self.labels[idx]
#         encoding = self.tokenizer(text, padding='max_length', max_length=self.max_length, truncation=True, return_attention_mask=True)
#         return {
#             'input_ids': torch.tensor(encoding['input_ids']),
#             'attention_mask': torch.tensor(encoding['attention_mask']),
#             'labels': torch.tensor(labels)
#         }

# @step
# def train_model_step(texts: list, labels: list) -> str:
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
#     task = 'sentiment'
#     MODEL = f"cardiffnlp/twitter-roberta-base-{task}"
#     tokenizer = AutoTokenizer.from_pretrained(MODEL)
#     model = AutoModelForSequenceClassification.from_pretrained(MODEL, num_labels=len(label_mapping)).to(device)

#     dataset = TweetsDataset(texts, labels, tokenizer)
#     train_loader = DataLoader(dataset, batch_size=16, shuffle=True)

#     optimizer = AdamW(model.parameters(), lr=5e-5)

#     model.train()
#     for epoch in range(2):  # loop over the dataset multiple times
#         total_loss = 0.0
#         for batch in tqdm(train_loader, desc=f"Training Epoch {epoch + 1}"):
#             batch = {k: v.to(device) for k, v in batch.items()}
#             outputs = model(**batch)
            
#             loss = outputs.loss
#             total_loss += loss.item()
            
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()
            
#         print(f"Epoch {epoch + 1}, Loss: {total_loss / len(train_loader)}")

#     # Save the trained model
#     model_path = "trained_sentiment_model"
#     model.save_pretrained(model_path)
#     tokenizer.save_pretrained(model_path)

#     return model_path


# In[34]:


@step
def model_inference_step(texts: list) -> list:
    predictions = []

    # Set the model path to the trained model directory
    model_path = 'trained_sentiment_model'
    
    # Load tokenizer and model from your trained model directory
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)

    # Assuming you have label mappings like before in the `train_model_step`
    # If they are stored differently for the new model, adjust as necessary.
    label_mapping = {'negative': 0, 'neutral': 1, 'positive': 2}
    labels = list(label_mapping.keys())

    for text in texts:
        encoded_input = tokenizer(text, return_tensors='pt')
        output = model(**encoded_input)
        scores = output[0][0].detach().numpy()
        scores = softmax(scores)
        ranking = np.argsort(scores)[::-1]
        text_predictions = [labels[i] for i in ranking] 
        predictions.append(text_predictions[0])

    return predictions



# In[35]:


@step
def evaluate_predictions(predictions: list, true_labels: list) -> dict:
    predictions_mapped = [label_mapping[pred] for pred in predictions]
    
    accuracy = accuracy_score(true_labels, predictions_mapped)
    precision = precision_score(true_labels, predictions_mapped, average='weighted', zero_division=0)
    recall = recall_score(true_labels, predictions_mapped, average='weighted', zero_division=0)
    f1 = f1_score(true_labels, predictions_mapped, average='weighted', zero_division=0)
    
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1_score", f1)
    
    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}


# In[36]:


import matplotlib.pyplot as plt


@step
def visualize_metrics(metrics: dict) -> str:
    
    names = list(metrics.keys())
    values = list(metrics.values())
    
    plt.figure(figsize=(10, 5))
    plt.bar(names, values)
    plt.ylabel('Score')
    plt.title('Model Evaluation Metrics')
    
    figure_path = 'metrics_figure.png'
    plt.savefig(figure_path)
    plt.close()
    
    return figure_path


# In[37]:


@pipeline
def sentiment_analysis_pipeline_with_evaluation(file_path: str, labels_path: str):
    tweets = read_tweets_from_file(file_path)
    true_labels = read_labels_from_file(labels_path)
    processed_texts = preprocess_step(tweets)
    insert_preprocessed_tweets_into_cassandra(processed_texts)
    #train_model_step(processed_texts, true_labels)
    predictions = model_inference_step(processed_texts)
    evaluation_results = evaluate_predictions(predictions, true_labels)
    visualize_metrics(evaluation_results)
    

    

if __name__ == "__main__":
    file_path = '../val_text.txt' 
    labels_path = '../val_labels.txt' 
    sentiment_analysis_pipeline_with_evaluation(file_path, labels_path)
    


# In[38]:


# CASSANDRA_CLUSTER = ['localhost']
# KEYSPACE = 'mykeyspace'
# TABLE_NAME = 'preprocessed_tweets'    


# def fetch_and_print_preprocessed_tweets():
#     query = f"SELECT id, tweet_text FROM {TABLE_NAME}"
#     rows = session.execute(query)
    
#     for row in rows:
#         print(f"ID: {row.id}, Tweet: {row.tweet_text}")
# fetch_and_print_preprocessed_tweets()


# In[ ]:


import pytest


def test_read_tweets_from_file():
    tweets = read_tweets_from_file('../val_text.txt')
    assert isinstance(tweets, list)
    assert all(isinstance(tweet, str) for tweet in tweets)

def test_preprocess_step():
    test_tweets = [
        "@user1 this is a test! http://testurl.com",
        "Normal text without users or urls"
    ]
    expected_output = [
        "@user this is a test! http",
        "Normal text without users or urls"
    ]
    preprocessed = preprocess_step(test_tweets)
    assert preprocessed == expected_output

