from unittest.mock import MagicMock, patch

import pytest

from pipeline import (evaluate_predictions,
                      insert_preprocessed_tweets_into_cassandra,
                      model_inference_step, preprocess_step,
                      read_labels_from_file, read_tweets_from_file)


def test_read_labels_from_file():
    labels = read_labels_from_file('../val_labels.txt')
    assert isinstance(labels, list)
    assert all(isinstance(label, int) for label in labels)
    assert len(labels) > 0  

@patch('pipeline.mlflow.log_metric')  
def test_evaluate_predictions(mock_log):
    predictions = ['positive', 'negative', 'neutral']
    true_labels = [2, 0, 1]
    results = evaluate_predictions(predictions, true_labels)
    assert isinstance(results, dict)
    assert set(results.keys()) == {"accuracy", "precision", "recall", "f1"}
    assert all(isinstance(value, float) for value in results.values())


@patch('pipeline.Cluster.connect')
def test_insert_preprocessed_tweets_into_cassandra(mock_connect):
    mock_session = MagicMock()
    mock_connect.return_value = mock_session
    processed_texts = ["Test tweet 1 processed", "Test tweet 2 processed"]
    insert_preprocessed_tweets_into_cassandra(processed_texts)
    assert mock_session.execute.call_count == len(processed_texts)
    
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
    
def test_model_inference_step():
    test_tweets = [
        "This is a positive tweet",
        "This is a negative tweet",
        "This is a neutral tweet"
    ]
    predictions = model_inference_step(test_tweets)
    assert isinstance(predictions, list)
    assert all(prediction in ["positive", "negative", "neutral"] for prediction in predictions)
    assert len(predictions) == len(test_tweets)
