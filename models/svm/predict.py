import os
import pandas as pd
import numpy as np
from datasets import load_dataset
from huggingface_hub import hf_hub_download
import joblib
import sys
import json

# Add the project root directory to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.append(project_root)

from utils.svm_trainer import SVMTrainer  # Import SVMTrainer for pickle to resolve the class

# Emotion labels for GoEmotions
go_emotion_labels = [
    'admiration', 'amusement', 'anger', 'annoyance', 'approval', 'caring',
    'confusion', 'curiosity', 'desire', 'disappointment', 'disapproval',
    'disgust', 'embarrassment', 'excitement', 'fear', 'gratitude', 'grief',
    'joy', 'love', 'nervousness', 'optimism', 'pride', 'realization',
    'relief', 'remorse', 'sadness', 'surprise', 'neutral'
]

def load_movie_reviews():
    """Load and prepare movie review dataset"""
    # Load IMDB dataset
    dataset = load_dataset("imdb")
    
    # We'll use the test set for evaluation
    test_data = dataset["test"]
    
    # Convert to pandas DataFrame for easier manipulation
    df = pd.DataFrame({
        'text': test_data['text'],
        'label': test_data['label']  # 0 for negative, 1 for positive
    })
    
    # Take a sample for testing (can be adjusted)
    sample_size = 5000
    df = df.sample(n=min(sample_size, len(df)), random_state=42)
    
    return df

def get_sentiment_from_emotions(emotion_indices):
    """Convert emotion indices to overall sentiment (1 for positive, 0 for negative)"""
    emotions = [go_emotion_labels[i] for i in emotion_indices]
    
    # Define positive and negative emotions
    positive_emotions = {
        'admiration', 'amusement', 'approval', 'caring', 'curiosity', 'desire',
        'excitement', 'gratitude', 'joy', 'love', 'optimism', 'pride',
        'realization', 'relief'
    }
    
    negative_emotions = {
        'anger', 'annoyance', 'confusion', 'disappointment', 'disapproval',
        'disgust', 'embarrassment', 'fear', 'grief', 'nervousness', 'remorse',
        'sadness'
    }
    
    # Count positive and negative emotions
    pos_count = sum(1 for e in emotions if e in positive_emotions)
    neg_count = sum(1 for e in emotions if e in negative_emotions)
    
    # If more positive emotions, return positive sentiment
    if pos_count > neg_count:
        return 1
    # If more negative emotions, return negative sentiment
    elif neg_count > pos_count:
        return 0
    # If equal or only neutral emotions, return the most common emotion's sentiment
    else:
        if emotions:
            first_emotion = emotions[0]
            if first_emotion in positive_emotions:
                return 1
            elif first_emotion in negative_emotions:
                return 0
        return 1  # Default to positive if no clear sentiment

def main():
    # Download and load the model from Hugging Face
    print("Downloading SVM model from Hugging Face...")
    file_path = hf_hub_download(
        repo_id="BakshiSan/SVM_Baseline",
        filename="svm_pipeline.joblib"
    )
    svm_pipeline = joblib.load(file_path)
    
    # Load movie reviews
    print("Loading movie reviews...")
    movie_reviews = load_movie_reviews()
    
    # Make predictions
    print("\nMaking predictions...")
    predictions = []
    probabilities = []
    
    for text in movie_reviews['text']:
        # Get emotion predictions
        pred_emotions = svm_pipeline.predict([text])[0]
        predictions.append(pred_emotions)
        
        # Get probabilities if available
        if hasattr(svm_pipeline, "predict_proba"):
            probs = svm_pipeline.predict_proba([text])[0]
            probabilities.append(probs)
        else:
            probabilities.append(None)
    
    # Convert predictions to emotion names
    predicted_emotions = []
    for pred in predictions:
        emotions = [go_emotion_labels[i] for i, p in enumerate(pred) if p == 1]
        predicted_emotions.append(emotions)
    
    # Convert to sentiment
    predicted_sentiments = [get_sentiment_from_emotions(pred) for pred in predictions]
    
    # Create results DataFrame
    results_df = pd.DataFrame({
        'text': movie_reviews['text'],
        'actual_sentiment': movie_reviews['label'].map({0: 'negative', 1: 'positive'}),
        'predicted_sentiment': ['positive' if s == 1 else 'negative' for s in predicted_sentiments],
        'predicted_emotions': [', '.join(emotions) for emotions in predicted_emotions]
    })
    
    # Save results
    results_path = "models/svm/results/movie_review_emotions.csv"
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    results_df.to_csv(results_path, index=False)
    
    # Calculate and print metrics
    accuracy = np.mean([1 if a == p else 0 for a, p in zip(movie_reviews['label'], predicted_sentiments)])
    print(f"\nAccuracy: {accuracy:.4f}")
    
    # Print example predictions
    print("\nExample Predictions:")
    print("-" * 80)
    for i in range(min(5, len(results_df))):
        print(f"\nReview {i+1}:")
        print(f"Text: {results_df['text'].iloc[i][:200]}...")
        print(f"Actual Sentiment: {results_df['actual_sentiment'].iloc[i]}")
        print(f"Predicted Sentiment: {results_df['predicted_sentiment'].iloc[i]}")
        print(f"Predicted Emotions: {results_df['predicted_emotions'].iloc[i]}")
        print("-" * 80)
    
    print(f"\nResults saved to {results_path}")

    # Create notebook content
    notebook_content = {
        "cells": [
            # ... (all the cells from above)
        ],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {
                    "name": "ipython",
                    "version": 3
                },
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.8.0"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }

    # Save notebook
    with open('models/svm/svm_prediction_notebook.ipynb', 'w') as f:
        json.dump(notebook_content, f)

if __name__ == "__main__":
    main() 