import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer
import torch
from torch.utils.data import Dataset, DataLoader
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import re
from datasets import load_dataset

class EmotionDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.float)
        }

def preprocess_text(text):
    """Basic text preprocessing"""
    # Convert to lowercase
    text = text.lower()
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    # Remove special characters and numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def load_goemotions_hf():
    """Load GoEmotions dataset from Hugging Face"""
    # Load the dataset
    dataset = load_dataset("go_emotions", "raw")
    
    # Get emotion labels
    emotion_labels = [
        'admiration', 'amusement', 'anger', 'annoyance', 'approval', 'caring',
        'confusion', 'curiosity', 'desire', 'disappointment', 'disapproval',
        'disgust', 'embarrassment', 'excitement', 'fear', 'gratitude', 'grief',
        'joy', 'love', 'nervousness', 'optimism', 'pride', 'realization',
        'relief', 'remorse', 'sadness', 'surprise', 'neutral'
    ]
    
    # Process the data
    def process_example(example):
        # Convert labels to one-hot encoding
        one_hot = np.zeros(len(emotion_labels))
        for label in example['labels']:
            one_hot[label] = 1
        return {
            'text': preprocess_text(example['text']),
            'labels': one_hot
        }
    
    # Process all splits
    processed_dataset = dataset.map(process_example)
    
    # Convert to lists for easier handling
    train_texts = processed_dataset['train']['text']
    train_labels = processed_dataset['train']['labels']
    val_texts = processed_dataset['validation']['text']
    val_labels = processed_dataset['validation']['labels']
    test_texts = processed_dataset['test']['text']
    test_labels = processed_dataset['test']['labels']
    
    return {
        'train': {'texts': train_texts, 'labels': train_labels},
        'val': {'texts': val_texts, 'labels': val_labels},
        'test': {'texts': test_texts, 'labels': test_labels},
        'emotion_labels': emotion_labels
    }

def load_movie_reviews_data(file_path):
    """Load and preprocess movie reviews dataset"""
    df = pd.read_csv(file_path)
    texts = df['text'].apply(preprocess_text)
    # Assuming sentiment labels are in columns after 'text'
    sentiment_columns = [col for col in df.columns if col != 'text']
    labels = df[sentiment_columns].values
    return texts, labels

def load_twitter_data(file_path):
    """Load and preprocess Twitter dataset"""
    df = pd.read_csv(file_path)
    texts = df['text'].apply(preprocess_text)
    # Assuming emotion labels are in columns after 'text'
    emotion_columns = [col for col in df.columns if col != 'text']
    labels = df[emotion_columns].values
    return texts, labels

def create_data_loaders(texts, labels, tokenizer, batch_size=32, test_size=0.2, val_size=0.1):
    """Create train, validation, and test data loaders"""
    # Split into train and test
    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=test_size, random_state=42
    )
    
    # Split train into train and validation
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=val_size, random_state=42
    )
    
    # Create datasets
    train_dataset = EmotionDataset(X_train, y_train, tokenizer)
    val_dataset = EmotionDataset(X_val, y_val, tokenizer)
    test_dataset = EmotionDataset(X_test, y_test, tokenizer)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    return train_loader, val_loader, test_loader

def get_tokenizer(model_name):
    """Get appropriate tokenizer for the model"""
    if 'bert' in model_name.lower():
        return AutoTokenizer.from_pretrained('bert-base-uncased')
    elif 'roberta' in model_name.lower():
        return AutoTokenizer.from_pretrained('roberta-base')
    else:
        raise ValueError(f"Unsupported model: {model_name}") 