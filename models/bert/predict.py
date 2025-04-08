import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from utils.data_processing import get_tokenizer
import os

def load_model(model_path='models/bert/saved_models/final_model'):
    """Load a trained model and tokenizer"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load tokenizer
    tokenizer = get_tokenizer('bert')
    
    # Load model
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.to(device)
    model.eval()
    
    return model, tokenizer, device

def predict_emotions(text, model, tokenizer, device, threshold=0.5):
    """Predict emotions for a given text"""
    # Preprocess and tokenize the text
    inputs = tokenizer(
        text,
        add_special_tokens=True,
        max_length=128,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    ).to(device)
    
    # Get model predictions
    with torch.no_grad():
        outputs = model(**inputs)
        probabilities = torch.sigmoid(outputs.logits)
        predictions = (probabilities > threshold).cpu().numpy()[0]
    
    return predictions, probabilities.cpu().numpy()[0]

def main():
    # Emotion labels (same as in training)
    emotion_labels = [
        'admiration', 'amusement', 'anger', 'annoyance', 'approval', 'caring',
        'confusion', 'curiosity', 'desire', 'disappointment', 'disapproval',
        'disgust', 'embarrassment', 'excitement', 'fear', 'gratitude', 'grief',
        'joy', 'love', 'nervousness', 'optimism', 'pride', 'realization',
        'relief', 'remorse', 'sadness', 'surprise', 'neutral'
    ]
    
    # Load model
    print("Loading model...")
    model, tokenizer, device = load_model()
    
    # Interactive prediction loop
    print("\nEnter text to predict emotions (or 'quit' to exit):")
    while True:
        text = input("\nEnter text: ")
        if text.lower() == 'quit':
            break
            
        predictions, probabilities = predict_emotions(text, model, tokenizer, device)
        
        # Get predicted emotions
        predicted_emotions = [emotion_labels[i] for i, pred in enumerate(predictions) if pred]
        
        print("\nPredicted emotions:", predicted_emotions)
        print("Top 3 emotion probabilities:")
        top_indices = probabilities.argsort()[-3:][::-1]
        for idx in top_indices:
            print(f"{emotion_labels[idx]}: {probabilities[idx]:.4f}")

if __name__ == "__main__":
    main() 