#!/usr/bin/env python3
"""
roberta_emotion.py

Multi‑label emotion classification on GoEmotions using RoBERTa.

Usage:
    python roberta_emotion.py

Prerequisites:
    pip install datasets transformers torch scikit-learn numpy pandas tqdm joblib
"""

import os
import time
import joblib
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import (
    RobertaTokenizer,
    RobertaForSequenceClassification,
    AdamW,
    get_linear_schedule_with_warmup
)
from datasets import load_dataset
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import classification_report, f1_score, precision_recall_curve
from tqdm import tqdm

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


class GoEmotionsDataset(Dataset):
    """Dataset class for GoEmotions"""

    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        labels = self.labels[idx]

        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        # Squeeze to remove batch dimension added by tokenizer
        input_ids = encoding["input_ids"].squeeze()
        attention_mask = encoding["attention_mask"].squeeze()

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": torch.FloatTensor(labels)
        }


def load_and_preprocess():
    """Load and preprocess the GoEmotions dataset"""
    start_time = time.time()
    print("Loading dataset...")

    # Load GoEmotions
    ds = load_dataset("go_emotions")
    labels = ds["train"].features["labels"].feature.names
    num_labels = len(labels)

    # Extract texts and labels
    train_texts, train_labels = ds["train"]["text"], ds["train"]["labels"]
    val_texts, val_labels = ds["validation"]["text"], ds["validation"]["labels"]
    test_texts, test_labels = ds["test"]["text"], ds["test"]["labels"]

    # Binarize labels
    mlb = MultiLabelBinarizer(classes=list(range(num_labels)))
    y_train = mlb.fit_transform(train_labels)
    y_val = mlb.transform(val_labels)
    y_test = mlb.transform(test_labels)

    print(f"Dataset loaded and preprocessed in {time.time() - start_time:.2f} seconds")
    print(
        f"Train samples: {len(train_texts)}, Validation samples: {len(val_texts)}, Test samples: {len(test_texts)}")

    return train_texts, y_train, val_texts, y_val, test_texts, y_test, labels, mlb


def create_dataloaders(train_texts, y_train, val_texts, y_val, test_texts, y_test, tokenizer,
                       batch_size=16):
    """Create DataLoader objects for training and evaluation"""
    start_time = time.time()
    print("Creating dataloaders...")

    # Create datasets
    train_dataset = GoEmotionsDataset(train_texts, y_train, tokenizer)
    val_dataset = GoEmotionsDataset(val_texts, y_val, tokenizer)
    test_dataset = GoEmotionsDataset(test_texts, y_test, tokenizer)

    # Create dataloaders
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size)

    print(f"Dataloaders created in {time.time() - start_time:.2f} seconds")

    return train_dataloader, val_dataloader, test_dataloader


def train_epoch(model, dataloader, optimizer, scheduler, epoch):
    """Train model for one epoch"""
    model.train()
    total_loss = 0

    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}")
    for batch in progress_bar:
        optimizer.zero_grad()

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )

        loss = outputs.loss
        total_loss += loss.item()

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})

    return total_loss / len(dataloader)


def evaluate(model, dataloader):
    """Evaluate model on validation or test data"""
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

            logits = outputs.logits
            all_preds.append(logits.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    all_preds = np.vstack(all_preds)
    all_labels = np.vstack(all_labels)

    return all_preds, all_labels


def find_optimal_thresholds(val_preds, val_labels):
    """Find optimal threshold for each emotion label"""
    thresholds = []

    for i in range(val_labels.shape[1]):
        precision, recall, thresh = precision_recall_curve(val_labels[:, i], val_preds[:, i])
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        optimal_idx = np.argmax(f1)
        thresholds.append(thresh[optimal_idx])

    return np.array(thresholds)


def train_model(train_dataloader, val_dataloader, test_dataloader, num_labels, epochs=4):
    """Train RoBERTa model for multi-label classification"""
    # Initialize model
    model_name = "roberta-base"
    model = RobertaForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        problem_type="multi_label_classification"
    )
    model.to(device)

    # Training parameters
    optimizer = AdamW(model.parameters(), lr=2e-5)
    total_steps = len(train_dataloader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=total_steps
    )

    best_f1 = 0
    best_model = None
    best_thresholds = None

    start_time = time.time()
    print(f"Starting training for {epochs} epochs...")

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()

        # Train
        train_loss = train_epoch(model, train_dataloader, optimizer, scheduler, epoch)

        # Validate
        val_preds, val_labels = evaluate(model, val_dataloader)
        thresholds = find_optimal_thresholds(val_preds, val_labels)
        val_binary_preds = (val_preds >= thresholds).astype(int)

        # Calculate metrics
        val_micro_f1 = f1_score(val_labels, val_binary_preds, average="micro")
        val_macro_f1 = f1_score(val_labels, val_binary_preds, average="macro")

        epoch_time = time.time() - epoch_start

        print(f"Epoch {epoch}: Train Loss = {train_loss:.4f}, Val Micro-F1 = {val_micro_f1:.4f}, "
              f"Val Macro-F1 = {val_macro_f1:.4f}, Time = {epoch_time:.2f}s")

        # Save best model
        if val_micro_f1 > best_f1:
            best_f1 = val_micro_f1
            best_model = model.state_dict().copy()
            best_thresholds = thresholds
            print(f"New best model with Micro-F1 = {val_micro_f1:.4f}")

    total_time = time.time() - start_time
    print(f"Training completed in {total_time:.2f} seconds ({total_time / 60:.2f} minutes)")

    # Load best model for final evaluation
    model.load_state_dict(best_model)

    # Evaluate on test set
    print("Evaluating on test set...")
    test_preds, test_labels = evaluate(model, test_dataloader)
    test_binary_preds = (test_preds >= best_thresholds).astype(int)

    test_micro_f1 = f1_score(test_labels, test_binary_preds, average="micro")
    test_macro_f1 = f1_score(test_labels, test_binary_preds, average="macro")

    print(f"Test Micro-F1: {test_micro_f1:.4f}")
    print(f"Test Macro-F1: {test_macro_f1:.4f}")

    return model, best_thresholds


def save_model(model, tokenizer, thresholds, emotion_labels, mlb, model_dir="roberta_model"):
    """Save the trained model and components"""
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    # Save model and tokenizer
    model.save_pretrained(os.path.join(model_dir, "model"))
    tokenizer.save_pretrained(os.path.join(model_dir, "tokenizer"))

    # Save thresholds and other components
    joblib.dump(thresholds, os.path.join(model_dir, "thresholds.joblib"))
    joblib.dump(emotion_labels, os.path.join(model_dir, "emotion_labels.joblib"))
    joblib.dump(mlb, os.path.join(model_dir, "multilabel_binarizer.joblib"))

    # Save model info
    model_info = {
        "model_type": "roberta-base",
        "num_labels": len(emotion_labels),
        "problem_type": "multi_label_classification",
        "labels": emotion_labels
    }
    joblib.dump(model_info, os.path.join(model_dir, "model_info.joblib"))

    print(f"Model and components saved to {model_dir}/")


def predict_sample(text, model, tokenizer, thresholds, emotion_labels):
    """Test prediction on a sample text"""
    model.eval()

    # Tokenize
    inputs = tokenizer(
        text,
        max_length=128,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )

    # Move to device
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    # Predict
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits.cpu().numpy()[0]

    # Apply thresholds
    preds = (logits >= thresholds).astype(int)

    # Get predicted emotions
    predicted_emotions = [emotion_labels[i] for i, pred in enumerate(preds) if pred == 1]

    return predicted_emotions


def main():
    """Main function to train and save RoBERTa model"""
    # Start timer for the whole process
    total_start_time = time.time()

    # Load and preprocess data
    train_texts, y_train, val_texts, y_val, test_texts, y_test, emotion_labels, mlb = load_and_preprocess()

    # Initialize tokenizer
    print("Loading RoBERTa tokenizer...")
    tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

    # Create dataloaders
    batch_size = 16  # Adjust based on available GPU memory
    train_dataloader, val_dataloader, test_dataloader = create_dataloaders(
        train_texts, y_train, val_texts, y_val, test_texts, y_test, tokenizer, batch_size
    )

    # Train model
    model, thresholds = train_model(
        train_dataloader, val_dataloader, test_dataloader, num_labels=len(emotion_labels), epochs=4
    )

    # Save model and components
    save_model(model, tokenizer, thresholds, emotion_labels, mlb, model_dir="roberta_model")

    # Test on sample
    print("\nTesting model on sample texts:")
    samples = [
        "I'm so happy today! Everything is going great.",
        "This makes me angry and frustrated.",
        "I'm not sure how to feel about this news."
    ]

    for sample in samples:
        emotions = predict_sample(sample, model, tokenizer, thresholds, emotion_labels)
        print(f"\nText: {sample}")
        print(f"Predicted emotions: {emotions}")

    # Print total time
    total_time = time.time() - total_start_time
    print(f"\nTotal execution time: {total_time:.2f} seconds ({total_time / 60:.2f} minutes)")

    # Print expected run time for future runs
    print("\nExpected run times for future training:")
    print(f"Dataset loading: ~{(time.time() - total_start_time) / 60:.1f} minutes")
    print(f"Training (4 epochs): ~{total_time / 60:.1f} minutes on {device}")
    print("Note: Times will vary based on hardware, especially if using GPU vs CPU")


if __name__ == "__main__":
    main()