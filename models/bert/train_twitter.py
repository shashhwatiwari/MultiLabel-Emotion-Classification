import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification
from torch.optim import AdamW
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from utils.data_processing import load_twitter_data, get_tokenizer, TwitterDataset, preprocess_text
from utils.evaluation import evaluate_model_single_label, print_metrics

def train_bert_twitter():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load dataset
    print("Loading Twitter dataset...")
    dataset = load_twitter_data()
    num_labels = len(dataset['emotion_labels'])

    # Get tokenizer and model
    tokenizer = get_tokenizer('bert')
    model = AutoModelForSequenceClassification.from_pretrained(
        'bert-base-uncased',
        num_labels=num_labels,
        problem_type="single_label_classification"
    ).to(device)

    # Create datasets
    train_dataset = TwitterDataset(
        dataset['train']['texts'],
        dataset['train']['labels'],
        tokenizer,
        num_labels
    )
    val_dataset = TwitterDataset(
        dataset['val']['texts'],
        dataset['val']['labels'],
        tokenizer,
        num_labels
    )
    test_dataset = TwitterDataset(
        dataset['test']['texts'],
        dataset['test']['labels'],
        tokenizer,
        num_labels
    )

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)
    test_loader = DataLoader(test_dataset, batch_size=32)

    # Training setup
    optimizer = AdamW(model.parameters(), lr=2e-5)
    criterion = torch.nn.CrossEntropyLoss()

    # Training loop
    num_epochs = 3
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        model.train()
        total_loss = 0
        total_batches = len(train_loader)
        
        progress_bar = tqdm(train_loader, desc=f"Training Epoch {epoch + 1}")
        
        for batch_idx, batch in enumerate(progress_bar):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'avg_loss': f'{total_loss/(batch_idx+1):.4f}'
            })

        avg_loss = total_loss / total_batches
        print(f"\nEpoch {epoch + 1} Summary:")
        print(f"Average Training Loss: {avg_loss:.4f}")

        print("\nEvaluating on validation set...")
        val_metrics = evaluate_model_single_label(model, val_loader, device)
        print_metrics(val_metrics, prefix="Validation ")

    # Evaluate on test set
    test_metrics = evaluate_model_single_label(model, test_loader, device)
    print_metrics(test_metrics, prefix="Test ")

    # Save model
    os.makedirs('models/bert/saved_models', exist_ok=True)
    model.save_pretrained('models/bert/saved_models/twitter_model')
    
    return model, tokenizer, dataset['emotion_labels']

def predict_sentiment(text, model, tokenizer, sentiment_labels, device, threshold=0.2):
    """Predict sentiment for a given text with probability scores for all sentiments"""
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
    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)
        probabilities = torch.softmax(outputs.logits, dim=1)
        probabilities = probabilities.cpu().numpy()[0]
        
        # Get all sentiments above threshold
        predicted_sentiments = []
        for idx, prob in enumerate(probabilities):
            if prob > threshold:
                predicted_sentiments.append((sentiment_labels[idx], prob))
        
        # Sort by probability
        predicted_sentiments.sort(key=lambda x: x[1], reverse=True)
    
    return predicted_sentiments, probabilities

if __name__ == "__main__":
    # Train the model
    model, tokenizer, sentiment_labels = train_bert_twitter()
    
    # Example predictions
    test_texts = [
        "I'm so happy to see you! This is amazing!",
        "I'm really disappointed with how this turned out.",
        "I'm both excited and nervous about the upcoming event.",
        "This is just okay, nothing special.",
        "I'm worried about what might happen next."
    ]
    
    print("\nTesting predictions on sample texts:")
    for text in test_texts:
        predicted_sentiments, probabilities = predict_sentiment(
            text, model, tokenizer, sentiment_labels, device
        )
        print(f"\nText: {text}")
        print("Predicted sentiments (with probabilities):")
        for sentiment, prob in predicted_sentiments:
            print(f"{sentiment}: {prob:.4f}")
        print("\nAll sentiment probabilities:")
        for idx, prob in enumerate(probabilities):
            print(f"{sentiment_labels[idx]}: {prob:.4f}") 