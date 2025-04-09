import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
import numpy as np
from transformers import (
    DistilBertForSequenceClassification,
    DistilBertTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding
)
from datasets import load_dataset
from sklearn.metrics import f1_score

# Emotion labels
emotion_labels = [
    'admiration', 'amusement', 'anger', 'annoyance', 'approval', 'caring',
    'confusion', 'curiosity', 'desire', 'disappointment', 'disapproval',
    'disgust', 'embarrassment', 'excitement', 'fear', 'gratitude', 'grief',
    'joy', 'love', 'nervousness', 'optimism', 'pride', 'realization',
    'relief', 'remorse', 'sadness', 'surprise', 'neutral'
]

def compute_metrics(eval_pred):
    """Compute metrics for evaluation"""
    logits, labels = eval_pred

    # Convert logits to probabilities
    probs = torch.sigmoid(torch.tensor(logits))

    # Binarize predictions at 0.5
    preds = (probs > 0.5).int().numpy()
    labels = labels.astype(np.int32)  # Ensure labels are integers for comparison

    # Micro F1 score
    f1_micro = f1_score(labels, preds, average="micro")
    subset_acc = np.mean(np.all(preds == labels, axis=1))

    return {
        "f1_micro": f1_micro,
        "subset_accuracy": subset_acc
    }

class CustomDataCollator(DataCollatorWithPadding):
    def __call__(self, features):
        batch = super().__call__(features)
        if 'labels' in batch:
            # Ensure labels are float tensors
            batch['labels'] = batch['labels'].float()
        return batch

def train_bert_goemotions():
    """Train BERT model on GoEmotions dataset"""
    # Load dataset
    dataset = load_dataset("go_emotions", "simplified")
    
    # One-hot encode the 'labels' field
    def one_hot_labels(example):
        label_vector = [0] * 28
        for label_id in example["labels"]:
            label_vector[label_id] = 1
        example["labels"] = label_vector
        return example

    # Apply the one-hot encoding
    dataset = dataset.map(one_hot_labels)
    
    # Initialize tokenizer
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    
    # Tokenize function
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=128
        )
    
    # Tokenize datasets
    tokenized_datasets = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=["text", "id"]
    )
    
    # Convert labels to float for BCEWithLogitsLoss
    tokenized_datasets = tokenized_datasets.map(
        lambda x: {"labels": torch.FloatTensor(x["labels"])}
    )
    
    # Convert to torch format
    tokenized_datasets.set_format(
        type="torch",
        columns=['input_ids', 'attention_mask', 'labels'],
        output_all_columns=False
    )
    
    # Initialize model
    model = DistilBertForSequenceClassification.from_pretrained(
        "distilbert-base-uncased",
        num_labels=28,  # Number of emotions
        problem_type="multi_label_classification"
    )
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir="./results",
        eval_strategy="epoch",
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        learning_rate=2e-5,
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_micro"
    )
    
    # Initialize trainer with custom data collator
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        compute_metrics=compute_metrics,
        data_collator=CustomDataCollator(tokenizer)
    )
    
    # Train model
    print("Training model...")
    trainer.train()
    
    # Save model and tokenizer
    print("Saving model...")
    model_path = "models/bert/saved_models/final_model"
    os.makedirs(model_path, exist_ok=True)
    model.save_pretrained(model_path)
    tokenizer.save_pretrained(model_path)
    
    return model, tokenizer

if __name__ == "__main__":
    # Train the model
    model, tokenizer = train_bert_goemotions()
    
    # Move model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # Example predictions
    test_texts = [
        "I'm so happy to see you! This is amazing!",
        "I'm really disappointed with how this turned out.",
        "I'm both excited and nervous about the upcoming event."
    ]
    
    print("\nTesting predictions on sample texts:")
    for text in test_texts:
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
            probabilities = torch.sigmoid(outputs.logits)[0].cpu().numpy()  # move to CPU for numpy
        
        # Get emotions above threshold
        predicted_emotions = [
            emotion_labels[i] 
            for i, prob in enumerate(probabilities) 
            if prob > 0.3
        ]
        
        print(f"\nText: {text}")
        print("Predicted Emotions:")
        for emotion in predicted_emotions:
            print(f"- {emotion}")
            
        print("\nTop 3 Emotion Probabilities:")
        top_indices = np.argsort(probabilities)[-3:][::-1]
        for idx in top_indices:
            print(f"{emotion_labels[idx]}: {probabilities[idx]:.4f}")
