import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset
import pandas as pd
from sklearn.metrics import classification_report, accuracy_score, precision_recall_fscore_support, confusion_matrix
import os
from collections import Counter

# Emotion labels for GoEmotions
go_emotion_labels = [
    'admiration', 'amusement', 'anger', 'annoyance', 'approval', 'caring',
    'confusion', 'curiosity', 'desire', 'disappointment', 'disapproval',
    'disgust', 'embarrassment', 'excitement', 'fear', 'gratitude', 'grief',
    'joy', 'love', 'nervousness', 'optimism', 'pride', 'realization',
    'relief', 'remorse', 'sadness', 'surprise', 'neutral'
]

# Define emotion sentiment mapping
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

neutral_emotions = {'neutral', 'surprise'}

def get_sentiment_from_emotions(emotion_indices):
    """Convert emotion indices to overall sentiment (1 for positive, 0 for negative)"""
    emotions = [go_emotion_labels[i] for i in emotion_indices]
    
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

def load_model(model_repo="BakshiSan/BERT"):
    """Load model and tokenizer from Hugging Face"""
    tokenizer = AutoTokenizer.from_pretrained(model_repo)
    model = AutoModelForSequenceClassification.from_pretrained(model_repo)
    
    # Move to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    return model, tokenizer, device

def predict_emotions(text, model, tokenizer, device, threshold=0.3):
    """Predict emotions for a given text"""
    # Tokenize input
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=128
    )
    
    # Move inputs to same device as model
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Get predictions
    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)
        probabilities = torch.sigmoid(outputs.logits)[0].cpu().numpy()
    
    # Get emotions above threshold
    predicted_emotions = [
        i for i, prob in enumerate(probabilities) 
        if prob > threshold
    ]
    
    return predicted_emotions, probabilities

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
    
    # Take a sample of 5000 reviews for testing
    sample_size = 5000
    df = df.sample(n=min(sample_size, len(df)), random_state=42)
    
    return df

def calculate_sentiment_metrics(y_true, y_pred):
    """Calculate sentiment classification metrics"""
    # Convert predictions to sentiment (1 for positive, 0 for negative)
    y_pred_sentiment = [get_sentiment_from_emotions(pred) for pred in y_pred]
    
    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred_sentiment)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred_sentiment, average='binary'
    )
    
    # Calculate confusion matrix
    cm = confusion_matrix(y_true, y_pred_sentiment)
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': cm
    }

def evaluate_model(model, tokenizer, device, df, emotion_labels):
    """Evaluate model on movie reviews"""
    predictions = []
    probabilities_list = []
    
    for text in df['text']:
        pred_emotions, probs = predict_emotions(text, model, tokenizer, device)
        predictions.append(pred_emotions)
        probabilities_list.append(probs)
    
    # Convert predictions to binary matrix
    pred_matrix = np.zeros((len(df), len(emotion_labels)))
    for i, preds in enumerate(predictions):
        for pred in preds:
            pred_matrix[i, pred] = 1
    
    # Analyze results
    results = {
        'average_emotions_per_review': np.mean([len(pred) for pred in predictions]),
        'most_common_emotions': pd.Series([emotion_labels[i] for preds in predictions for i in preds]).value_counts().head(10),
        'prediction_matrix': pred_matrix,
        'probabilities': np.array(probabilities_list)
    }
    
    return results, predictions

def print_example_reviews(df, predictions, emotion_labels, num_examples=5):
    """Print example reviews and their predicted emotions"""
    print("\nExample Reviews and Their Predicted Emotions:")
    print("-" * 80)
    
    for i in range(min(num_examples, len(df))):
        review = df.iloc[i]
        pred_emotions = [emotion_labels[idx] for idx in predictions[i]]
        pred_sentiment = "Positive" if get_sentiment_from_emotions(predictions[i]) == 1 else "Negative"
        
        print(f"\nReview {i+1}:")
        print(f"Text: {review['text'][:200]}...")  # Print first 200 chars
        print(f"Actual Sentiment: {'Positive' if review['label'] == 1 else 'Negative'}")
        print(f"Predicted Sentiment: {pred_sentiment}")
        print(f"Predicted Emotions: {', '.join(pred_emotions)}")
        print("-" * 80)

def print_metrics(metrics):
    """Print evaluation metrics"""
    print("\nSentiment Classification Metrics:")
    print("-" * 80)
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1 Score: {metrics['f1']:.4f}")
    
    print("\nConfusion Matrix:")
    print("-" * 80)
    print("           Predicted")
    print("           Negative Positive")
    print(f"Actual Negative  {metrics['confusion_matrix'][0][0]:6d}  {metrics['confusion_matrix'][0][1]:6d}")
    print(f"        Positive  {metrics['confusion_matrix'][1][0]:6d}  {metrics['confusion_matrix'][1][1]:6d}")

def main():
    # Load GoEmotions model from Hugging Face
    print("Loading GoEmotions model from Hugging Face...")
    go_model, go_tokenizer, device = load_model()
    
    # Load movie reviews
    print("Loading movie reviews...")
    movie_reviews = load_movie_reviews()
    
    # Evaluate GoEmotions model
    print("\nEvaluating GoEmotions model on movie reviews...")
    go_results, predictions = evaluate_model(go_model, go_tokenizer, device, movie_reviews, go_emotion_labels)
    
    # Calculate sentiment metrics
    metrics = calculate_sentiment_metrics(movie_reviews['label'].values, predictions)
    
    # Print results
    print("\nGoEmotions Model Results:")
    print(f"Average emotions per review: {go_results['average_emotions_per_review']:.2f}")
    print("\nTop 10 most common emotions:")
    print(go_results['most_common_emotions'])
    
    # Print metrics
    print_metrics(metrics)
    
    # Print example reviews
    print_example_reviews(movie_reviews, predictions, go_emotion_labels)
    
    # Save results to CSV
    results_df = pd.DataFrame({
        'text': movie_reviews['text'],
        'actual_sentiment': movie_reviews['label'].map({0: 'negative', 1: 'positive'}),
        'predicted_sentiment': ['positive' if get_sentiment_from_emotions(pred) == 1 else 'negative' for pred in predictions],
        'predicted_emotions': [', '.join([go_emotion_labels[i] for i in preds]) for preds in predictions]
    })
    
    results_path = "models/bert/results/movie_review_emotions.csv"
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    results_df.to_csv(results_path, index=False)
    print(f"\nResults saved to {results_path}")

if __name__ == "__main__":
    main() 