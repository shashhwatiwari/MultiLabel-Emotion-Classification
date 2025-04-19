import os
import pandas as pd
import numpy as np
from datasets import load_dataset
from sklearn.preprocessing import MultiLabelBinarizer
import sys
import re
import emoji
import html
sys.path.append('../../')
from utils.svm_trainer import SVMTrainer

# Emotion labels for Twitter dataset
twitter_emotion_labels = [
    'anger', 'anticipation', 'disgust', 'fear', 'joy', 'love',
    'optimism', 'pessimism', 'sadness', 'surprise', 'trust'
]

def clean_tweet(text):
    """Clean Twitter text data"""
    # Convert HTML entities
    text = html.unescape(text)
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    
    # Remove mentions
    text = re.sub(r'@\w+', '', text)
    
    # Remove hashtags (but keep the word)
    text = re.sub(r'#(\w+)', r'\1', text)
    
    # Convert emojis to text
    text = emoji.demojize(text)
    
    # Remove special characters and numbers
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\d+', '', text)
    
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    # Convert to lowercase
    text = text.lower()
    
    return text

def load_twitter_data():
    """Load and prepare Twitter emotion dataset"""
    # Load dataset
    dataset = load_dataset("dair-ai/emotion")
    
    # Convert to pandas DataFrame
    train_data = dataset["train"]
    test_data = dataset["test"]
    
    # Convert to DataFrame
    train_df = pd.DataFrame({
        'text': train_data['text'],
        'labels': train_data['label']
    })
    
    test_df = pd.DataFrame({
        'text': test_data['text'],
        'labels': test_data['label']
    })
    
    # Clean text data
    print("Cleaning text data...")
    train_df['text'] = train_df['text'].apply(clean_tweet)
    test_df['text'] = test_df['text'].apply(clean_tweet)
    
    return train_df, test_df

def prepare_labels(labels_list):
    """Convert list of label indices to binary format"""
    mlb = MultiLabelBinarizer(classes=range(len(twitter_emotion_labels)))
    return mlb.fit_transform(labels_list)

def main():
    # Create output directory
    output_dir = "models/svm/saved_models/twitter"
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data
    print("Loading Twitter emotion dataset...")
    train_df, test_df = load_twitter_data()
    
    # Prepare labels
    print("Preparing labels...")
    y_train = prepare_labels(train_df['labels'])
    y_test = prepare_labels(test_df['labels'])
    
    # Initialize SVM trainer
    print("Initializing SVM trainer...")
    trainer = SVMTrainer(
        ngram_range=(1, 2),
        max_df=0.9,
        min_df=5,
        C=1.0,
        max_iter=10000,
        is_multilabel=True
    )
    
    # Split training data into train and validation
    X_train, X_val, _, y_train, y_val, _ = trainer.prepare_data(
        train_df['text'].tolist(),
        y_train,
        test_size=0.2,
        val_size=0.1
    )
    
    # Tune hyperparameters
    print("\nTuning hyperparameters...")
    param_grid = {
        'vectorizer__ngram_range': [(1, 1), (1, 2)],
        'vectorizer__min_df': [1, 5, 10],
        'classifier__estimator__C': [0.1, 1, 10]
    }
    best_params = trainer.tune_hyperparameters(
        X_train,
        y_train,
        param_grid=param_grid,
        cv=3,  # Using 3-fold CV for speed
        scoring='f1_macro'
    )
    
    # Train model
    print("\nTraining model...")
    trainer.train(X_train, y_train, X_val, y_val)
    
    # Evaluate on test set
    print("\nEvaluating on test set...")
    metrics = trainer.evaluate(
        test_df['text'].tolist(),
        y_test,
        target_names=twitter_emotion_labels
    )
    
    # Save model
    print("\nSaving model...")
    trainer.save_model(output_dir)
    
    # Save evaluation metrics
    metrics_df = pd.DataFrame(metrics['classification_report']).transpose()
    metrics_df.to_csv(os.path.join(output_dir, 'evaluation_metrics.csv'))
    
    print("\nTraining complete!")
    print(f"Model saved to: {output_dir}")
    print("\nTest Set Performance:")
    print(metrics['classification_report'])

if __name__ == "__main__":
    main() 