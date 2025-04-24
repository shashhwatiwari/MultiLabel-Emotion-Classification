from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Replace "your-username/model-repo-name" with the actual path to your model on HF.
model_repo = "BakshiSan/distilbert_GoEmo"  # e.g. "BakshiSan/BERT”
tokenizer = AutoTokenizer.from_pretrained(model_repo)
model = AutoModelForSequenceClassification.from_pretrained(model_repo)

# Optionally, move the model to the appropriate device (GPU, CPU, etc.)
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
def predict_emotions(text, threshold=0.5):
    model.eval()  # Set model to evaluation mode
    # Tokenize the text
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding="max_length", max_length=128)
    inputs = {key: value.to(device) for key, value in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.sigmoid(logits)[0].cpu().numpy()
    
    label_names = [
        'admiration', 'amusement', 'anger', 'annoyance', 'approval', 'caring',
        'confusion', 'curiosity', 'desire', 'disappointment', 'disapproval',
        'disgust', 'embarrassment', 'excitement', 'fear', 'gratitude', 'grief',
        'joy', 'love', 'nervousness', 'optimism', 'pride', 'realization',
        'relief', 'remorse', 'sadness', 'surprise', 'neutral'
    ]  # List of emotion names in order.
    
    # Get predicted emotions that exceed the threshold
    predicted_emotions = [label_names[i] for i, prob in enumerate(probs) if prob > threshold]
    
    return predicted_emotions, probs

# Example usage:
text_input = "i love this product! It's amazing."
emotions, probabilities = predict_emotions(text_input)
print("Predicted Emotions:", emotions)