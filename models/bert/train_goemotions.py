import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification
from torch.optim import AdamW
from utils.data_processing import load_goemotions_hf, get_tokenizer, EmotionDataset
from utils.evaluation import evaluate_model, print_metrics
from tqdm import tqdm

def predict_emotions(text, model, tokenizer, emotion_labels, device, threshold=0.5):
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
    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)
        probabilities = torch.sigmoid(outputs.logits)
        predictions = (probabilities > threshold).cpu().numpy()[0]
    
    # Get predicted emotions
    predicted_emotions = [emotion_labels[i] for i, pred in enumerate(predictions) if pred]
    return predicted_emotions, probabilities.cpu().numpy()[0]

def train_bert():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load dataset
    print("Loading dataset...")
    dataset = load_goemotions_hf()
    num_labels = len(dataset['emotion_labels'])

    # Get tokenizer and model
    tokenizer = get_tokenizer('bert')
    model = AutoModelForSequenceClassification.from_pretrained(
        'bert-base-uncased',
        num_labels=num_labels,
        problem_type="multi_label_classification"
    ).to(device)

    # Create datasets
    train_dataset = EmotionDataset(
        dataset['train']['texts'],
        dataset['train']['labels'],
        tokenizer,
        num_labels
    )
    val_dataset = EmotionDataset(
        dataset['val']['texts'],
        dataset['val']['labels'],
        tokenizer,
        num_labels
    )
    test_dataset = EmotionDataset(
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
    criterion = torch.nn.BCEWithLogitsLoss()

    # Training loop
    num_epochs = 3
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        model.train()
        total_loss = 0
        total_batches = len(train_loader)
        
        # Add progress bar
        progress_bar = tqdm(train_loader, desc=f"Training Epoch {epoch + 1}")
        
        for batch_idx, batch in enumerate(progress_bar):
            # Move batch to device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            # Forward pass
            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            loss = criterion(logits, labels)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'avg_loss': f'{total_loss/(batch_idx+1):.4f}'
            })

        # Print epoch summary
        avg_loss = total_loss / total_batches
        print(f"\nEpoch {epoch + 1} Summary:")
        print(f"Average Training Loss: {avg_loss:.4f}")

        # Evaluate on validation set
        print("\nEvaluating on validation set...")
        val_metrics = evaluate_model(model, val_loader, device)
        print_metrics(val_metrics, prefix="Validation ")

    # Evaluate on test set
    test_metrics = evaluate_model(model, test_loader, device)
    print_metrics(test_metrics, prefix="Test ")

    # Save model
    os.makedirs('models/bert/saved_models', exist_ok=True)
    model.save_pretrained('models/bert/saved_models/final_model')
    
    return model, tokenizer, dataset['emotion_labels']

if __name__ == "__main__":
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Train the model
    model, tokenizer, emotion_labels = train_bert()
    
    # Example predictions
    test_texts = [
        "I'm so happy to see you! This is amazing!",
        "I'm really disappointed with how this turned out.",
        "I'm both excited and nervous about the upcoming event."
    ]
    
    print("\nTesting predictions on sample texts:")
    for text in test_texts:
        predicted_emotions, probabilities = predict_emotions(
            text, model, tokenizer, emotion_labels, device
        )
        print(f"\nText: {text}")
        print("Predicted emotions:", predicted_emotions)
        print("Top 3 emotion probabilities:")
        top_indices = probabilities.argsort()[-3:][::-1]
        for idx in top_indices:
            print(f"{emotion_labels[idx]}: {probabilities[idx]:.4f}") 