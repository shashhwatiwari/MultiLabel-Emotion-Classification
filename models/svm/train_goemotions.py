import os
import pandas as pd
import numpy as np
from datasets import load_dataset
from sklearn.preprocessing import MultiLabelBinarizer
import sys
sys.path.append('../../')
from utils.svm_trainer import SVMTrainer

# Emotion labels for GoEmotions
go_emotion_labels = [
    'admiration', 'amusement', 'anger', 'annoyance', 'approval', 'caring',
    'confusion', 'curiosity', 'desire', 'disappointment', 'disapproval',
    'disgust', 'embarrassment', 'excitement', 'fear', 'gratitude', 'grief',
    'joy', 'love', 'nervousness', 'optimism', 'pride', 'realization',
    'relief', 'remorse', 'sadness', 'surprise', 'neutral'
]

def load_goemotions_data():
    """Load and prepare GoEmotions dataset"""
    # Load dataset
    dataset = load_dataset("go_emotions")
    
    # Convert to pandas DataFrame
    train_data = dataset["train"]
    test_data = dataset["test"]
    
    # Convert to DataFrame
    train_df = pd.DataFrame({
        'text': train_data['text'],
        'labels': train_data['labels']
    })
    
    test_df = pd.DataFrame({
        'text': test_data['text'],
        'labels': test_data['labels']
    })
    
    return train_df, test_df

def prepare_labels(labels_list):
    """Convert list of label indices to binary format"""
    mlb = MultiLabelBinarizer(classes=range(len(go_emotion_labels)))
    return mlb.fit_transform(labels_list)

def main():
    # Create output directory
    output_dir = "models/svm/saved_models/go_emotions"
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data
    print("Loading GoEmotions dataset...")
    train_df, test_df = load_goemotions_data()
    
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
        target_names=go_emotion_labels
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