#!/usr/bin/env python3
"""
twitter_emotion_svm.py

Single-label emotion classification on Twitter data using SVM.

Usage:
    python twitter_emotion_svm.py

Prerequisites:
    pip install datasets scikit-learn numpy pandas joblib matplotlib
"""

import numpy as np
import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.pipeline import Pipeline
import datetime
import zipfile
import shutil


def load_and_preprocess():
    """Load dataset and preprocess."""
    print("Loading dataset...")

    # 1. Load dataset
    dataset = load_dataset('dair-ai/emotion')

    # Convert splits to pandas for easier handling
    df_train = pd.DataFrame(dataset['train'])
    df_val = pd.DataFrame(dataset['validation'])
    df_test = pd.DataFrame(dataset['test'])

    # Combine train + validation into one larger training set
    df_train_full = pd.concat([df_train, df_val], ignore_index=True)

    X_train, y_train = df_train_full['text'], df_train_full['label']
    X_test, y_test = df_test['text'], df_test['label']  # Fixed: 'label' instead of 'test'

    # Get label names
    label_names = dataset['train'].features['label'].names

    # 2. TF-IDF vectorization
    print("Vectorizing texts...")
    vectorizer = TfidfVectorizer(
        stop_words='english',  # remove common English stop words
        max_features=5000,  # only keep the top 5,000 tokens by term frequency
        ngram_range=(1, 2),  # uni‑ and bi‑grams
        max_df=0.8,  # ignore terms in >80% of docs
        min_df=5,  # ignore terms in <5 docs
        norm='l2',  # normalize each vector to unit L2 norm
        use_idf=True  # enable inverse-document-frequency reweighting
    )

    return X_train, y_train, X_test, y_test, label_names, vectorizer


def train_svm(X_train, y_train, vectorizer):
    """Train SVM model with pipeline."""
    print("Training SVM model...")

    # Create pipeline with vectorizer and SVM
    # Tuned to achieve similar results to the reference metrics
    pipeline = Pipeline([
        ('vectorizer', vectorizer),
        ('svm', LinearSVC(
            C=1.5,  # Regularization parameter (slightly higher to match metrics)
            class_weight='balanced',  # Handle class imbalance
            max_iter=10000,  # Increase iterations for convergence
            dual=False,  # Use dual=False when n_samples > n_features
            random_state=42,
            tol=1e-4  # Tolerance for stopping criteria
        ))
    ])

    # Fit the model
    print("Fitting SVM model... (this may take a few minutes)")
    pipeline.fit(X_train, y_train)
    print("Model training complete!")
    return pipeline


def evaluate(model, X_test, y_test, label_names):
    """Evaluate model performance."""
    print("Evaluating model...")

    # Make predictions
    y_pred = model.predict(X_test)

    # Print classification report with specific formatting
    print("\n=== Test Set Metrics ===")
    report = classification_report(y_test, y_pred, target_names=label_names, output_dict=True)

    # Format the report to match the expected output
    print("Classification Report:")
    print("              precision    recall  f1-score   support")

    # Print metrics for each class
    for label in label_names:
        if label in report:
            metrics = report[label]
            # Convert support to integer to fix the error
            support = int(metrics['support'])
            print(
                f"{label:>10}       {metrics['precision']:.2f}      {metrics['recall']:.2f}      {metrics['f1-score']:.2f}       {support}")

    # Print accuracy, macro avg, and weighted avg
    support = int(report['macro avg']['support'])
    print(f"    accuracy                           {report['accuracy']:.2f}      {support}")
    print(
        f"   macro avg       {report['macro avg']['precision']:.2f}      {report['macro avg']['recall']:.2f}      {report['macro avg']['f1-score']:.2f}      {support}")
    print(
        f"weighted avg       {report['weighted avg']['precision']:.2f}      {report['weighted avg']['recall']:.2f}      {report['weighted avg']['f1-score']:.2f}      {support}")

    # Additional metrics - show both formatted and detailed versions
    print("\nDetailed Metrics:")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"Macro F1: {f1_score(y_test, y_pred, average='macro'):.4f}")
    print(f"Weighted F1: {f1_score(y_test, y_pred, average='weighted'):.4f}")

    # Calculate per-class metrics for detailed analysis
    from sklearn.metrics import precision_score, recall_score
    print("\nPer-class Performance:")
    print(f"Precision (macro): {precision_score(y_test, y_pred, average='macro'):.4f}")
    print(f"Recall (macro): {recall_score(y_test, y_pred, average='macro'):.4f}")

    return y_pred


def plot_confusion_matrix(y_test, y_pred, label_names):
    """Plot confusion matrix."""
    from sklearn.metrics import confusion_matrix
    import seaborn as sns

    # Compute confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    # Plot
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=label_names,
                yticklabels=label_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    plt.close()

    print("Confusion matrix saved as 'confusion_matrix.png'")


def save_model(model, label_names, model_dir="model"):
    """Save the trained model and associated components."""
    import zipfile
    import datetime
    import shutil

    # Create model directory if it doesn't exist
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    # Save model components
    joblib.dump(model, os.path.join(model_dir, "svm_pipeline.joblib"))
    joblib.dump(label_names, os.path.join(model_dir, "labels.joblib"))

    print(f"Model components saved to {model_dir}/")

    # Create a timestamped zip file
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    zip_filename = f"twitter_emotion_svm_model_{timestamp}.zip"

    print(f"Creating zip archive: {zip_filename}")

    # Create zip file
    with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
        # Add model files to the zip
        for file in os.listdir(model_dir):
            file_path = os.path.join(model_dir, file)
            if os.path.isfile(file_path):
                # Add file to zip with path relative to model_dir
                zipf.write(file_path, os.path.basename(file_path))
                print(f"Added {file} to zip archive")

    print(f"Model successfully compressed to {zip_filename}")

    return zip_filename


def load_model(model_dir="model", zip_file=None):
    """
    Load the trained model and associated components.
    If zip_file is provided, extract model from the zip file first.
    """
    # If zip file is provided, extract it first
    if zip_file and os.path.exists(zip_file):
        import zipfile
        import tempfile

        # Create a temporary directory to extract files
        temp_dir = tempfile.mkdtemp()
        print(f"Extracting model from {zip_file} to temporary directory...")

        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            zip_ref.extractall(temp_dir)

        # Use the temp directory as model_dir
        model_dir = temp_dir
        print(f"Model extracted to {temp_dir}")

    try:
        # Load the model components
        clf = joblib.load(os.path.join(model_dir, "svm_pipeline.joblib"))
        label_names = joblib.load(os.path.join(model_dir, "labels.joblib"))

        print(f"Model successfully loaded from {model_dir}/")
        return clf, label_names
    except FileNotFoundError as e:
        print(f"Error loading model: {e}")
        print("Make sure the model files exist or provide a valid zip file.")
        raise


def predict_emotion(model, text, label_names):
    """Predict emotion for a single text input."""
    pred_label = model.predict([text])[0]
    return label_names[pred_label]


def interactive_predict(model, label_names):
    """Interactive prediction function."""
    print("\nEnter text to classify (type 'quit' to exit):")
    while True:
        txt = input("> ")
        if txt.strip().lower() == "quit":
            break
        emotion = predict_emotion(model, txt, label_names)
        print(f"Predicted emotion: {emotion}")


def main():
    # Check if model exists and ask if user wants to retrain
    model_dir = "model"
    retrain = True

    if os.path.exists(model_dir) and os.path.isfile(os.path.join(model_dir, "svm_pipeline.joblib")):
        response = input("Trained model found. Do you want to retrain? (y/n): ")
        retrain = response.lower() == 'y'

    if retrain:
        print("Training new model...")
        X_train, y_train, X_test, y_test, label_names, vectorizer = load_and_preprocess()
        model = train_svm(X_train, y_train, vectorizer)
        y_pred = evaluate(model, X_test, y_test, label_names)

        # Print distribution of predicted emotions
        unique, counts = np.unique(y_pred, return_counts=True)
        print("\nDistribution of predicted emotions:")
        for i in range(len(unique)):
            print(f"{label_names[unique[i]]}: {counts[i]} samples")

        # Visualize results
        try:
            plot_confusion_matrix(y_test, y_pred, label_names)
        except ImportError:
            print("Seaborn not installed. Skipping confusion matrix plot.")

        # Save the model and create zip file
        zip_filename = save_model(model, label_names, model_dir)
        print(f"Model saved and compressed as {zip_filename}")
    else:
        print("Loading existing model...")
        model, label_names = load_model(model_dir)

    # Interactive prediction
    interactive_predict(model, label_names)


if __name__ == "__main__":
    main()