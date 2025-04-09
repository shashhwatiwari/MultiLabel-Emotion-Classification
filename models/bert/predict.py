import torch
import numpy as np
from transformers import DistilBertForSequenceClassification, DistilBertTokenizer
from datasets import load_dataset
import pandas as pd
from sklearn.metrics import classification_report
import os

# Emotion labels for GoEmotions
go_emotion_labels = [
    'admiration', 'amusement', 'anger', 'annoyance', 'approval', 'caring',
    'confusion', 'curiosity', 'desire', 'disappointment', 'disapproval',
    'disgust', 'embarrassment', 'excitement', 'fear', 'gratitude', 'grief',
    'joy', 'love', 'nervousness', 'optimism', 'pride', 'realization',
    'relief', 'remorse', 'sadness', 'surprise', 'neutral'
]

def load_model(model_path, tokenizer_path):
    """Load a saved model and tokenizer"""
    model = DistilBertForSequenceClassification.from_pretrained(model_path)
    tokenizer = DistilBertTokenizer.from_pretrained(tokenizer_path)
    
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
    
    # Take a sample for testing (can be adjusted)
    sample_size = 1000
    df = df.sample(n=min(sample_size, len(df)), random_state=42)
    
    return df

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
    
    return results

def main():
    # Load GoEmotions model
    go_emotions_model_path = "models/bert/saved_models/go_emotions/go_emotions_model"
    go_emotions_tokenizer_path = "models/bert/saved_models/go_emotions/go_emotions_tokenizer"
    
    print("Loading GoEmotions model...")
    go_model, go_tokenizer, device = load_model(go_emotions_model_path, go_emotions_tokenizer_path)
    
    # Load movie reviews
    print("Loading movie reviews...")
    movie_reviews = load_movie_reviews()
    
    # Evaluate GoEmotions model
    print("\nEvaluating GoEmotions model on movie reviews...")
    go_results = evaluate_model(go_model, go_tokenizer, device, movie_reviews, go_emotion_labels)
    
    # Print results
    print("\nGoEmotions Model Results:")
    print(f"Average emotions per review: {go_results['average_emotions_per_review']:.2f}")
    print("\nTop 10 most common emotions:")
    print(go_results['most_common_emotions'])
    
    # Save results to CSV
    results_df = pd.DataFrame({
        'text': movie_reviews['text'],
        'sentiment': movie_reviews['label'].map({0: 'negative', 1: 'positive'}),
        'predicted_emotions': [', '.join([go_emotion_labels[i] for i in preds]) for preds in go_results['prediction_matrix'].argmax(axis=1)]
    })
    
    results_path = "models/bert/results/movie_review_emotions.csv"
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    results_df.to_csv(results_path, index=False)
    print(f"\nResults saved to {results_path}")

if __name__ == "__main__":
    main() 