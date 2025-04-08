import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.data_processing import load_goemotions_hf, get_tokenizer, EmotionDataset
from utils.evaluation import print_metrics
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification

def main():
    # Load the dataset
    print("Loading GoEmotions dataset...")
    dataset = load_goemotions_hf()
    
    # Print dataset statistics
    print("\nDataset Statistics:")
    print(f"Number of training examples: {len(dataset['train']['texts'])}")
    print(f"Number of validation examples: {len(dataset['val']['texts'])}")
    print(f"Number of test examples: {len(dataset['test']['texts'])}")
    print(f"Number of emotion labels: {len(dataset['emotion_labels'])}")
    print("\nEmotion labels:", dataset['emotion_labels'])
    
    # Example: Print first few training examples
    print("\nFirst few training examples:")
    for i in range(3):
        print(f"\nExample {i+1}:")
        print(f"Text: {dataset['train']['texts'][i]}")
        print("Labels:", [dataset['emotion_labels'][j] for j, val in enumerate(dataset['train']['labels'][i]) if val == 1])
    
    # Example: Create data loaders for BERT
    print("\nCreating data loaders for BERT...")
    tokenizer = get_tokenizer('bert')
    
    # Create datasets
    train_dataset = EmotionDataset(
        dataset['train']['texts'],
        dataset['train']['labels'],
        tokenizer
    )
    
    # Create data loader
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    
    # Example: Load a batch
    print("\nLoading a batch of data:")
    batch = next(iter(train_loader))
    print(f"Batch input_ids shape: {batch['input_ids'].shape}")
    print(f"Batch attention_mask shape: {batch['attention_mask'].shape}")
    print(f"Batch labels shape: {batch['labels'].shape}")

if __name__ == "__main__":
    main() 