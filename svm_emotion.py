#!/usr/bin/env python3
"""
svm_emotion.py

Multi‑label emotion classification on GoEmotions using SVM.

Usage:
    python svm_emotion.py

Prerequisites:
    pip install datasets scikit-learn imbalanced-learn numpy joblib
"""

import numpy as np
import os
import joblib
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    f1_score,
    precision_recall_curve,
)
from imblearn.over_sampling import RandomOverSampler
from sklearn.pipeline import Pipeline as SkPipeline
from imblearn.pipeline import Pipeline as ImbPipeline


def load_and_preprocess():
    # 1. Load dataset
    ds = load_dataset("go_emotions")
    labels = ds["train"].features["labels"].feature.names

    # 2. Split texts & multi‑label lists
    train_texts, train_labels = ds["train"]["text"], ds["train"]["labels"]
    val_texts, val_labels = ds["validation"]["text"], ds["validation"]["labels"]

    # 3. Binarize labels (shape: n_samples × n_classes)
    mlb = MultiLabelBinarizer(classes=list(range(len(labels))))
    y_train = mlb.fit_transform(train_labels)
    y_val = mlb.transform(val_labels)

    # 4. TF–IDF vectorization
    vect = TfidfVectorizer(
        max_features=10_000,
        ngram_range=(1, 2),
        stop_words="english"
    )
    X_train = vect.fit_transform(train_texts)
    X_val = vect.transform(val_texts)

    return X_train, y_train, X_val, y_val, labels, vect, mlb


def train_svm(X_train, y_train):
    """
    Constructs a One‑vs‑Rest pipeline where each binary task
    first oversamples its positive/negative splits, then fits
    a LinearSVC with class_weight='balanced'.
    """
    # Inner pipeline for *each* binary problem:
    inner = ImbPipeline([
        ("oversample", RandomOverSampler(random_state=42)),
        ("svc", LinearSVC(class_weight="balanced", max_iter=10000))
    ])

    ovr = OneVsRestClassifier(inner, n_jobs=-1)
    ovr.fit(X_train, y_train)
    return ovr


def tune_thresholds(clf, X_val, y_val):
    # For each class, find threshold on validation F1
    scores = clf.decision_function(X_val)
    best_thr = []
    for i in range(y_val.shape[1]):
        p, r, thr = precision_recall_curve(y_val[:, i], scores[:, i])
        f1 = 2 * p * r / (p + r + 1e-8)
        best_thr.append(thr[np.argmax(f1)])
    return np.array(best_thr)


from sklearn.metrics import hamming_loss, accuracy_score


def evaluate(clf, X_val, y_val, thr, labels):
    scores = clf.decision_function(X_val)
    preds = (scores >= thr).astype(int)

    print("=== Validation Metrics ===")
    print(classification_report(y_val, preds, target_names=labels))

    # Existing metrics
    print(f"Raw accuracy : {accuracy_score(y_val, preds):.4f}")
    print(f"Micro‑F1      : {f1_score(y_val, preds, average='micro'):.4f}")
    print(f"Macro‑F1      : {f1_score(y_val, preds, average='macro'):.4f}")
    print(f"Weighted‑F1   : {f1_score(y_val, preds, average='weighted'):.4f}")

    # Additional metrics you requested
    print(f"Subset accuracy  : {accuracy_score(y_val, preds):.4f}")  # Same as raw accuracy
    print(f"Hamming accuracy : {1 - hamming_loss(y_val, preds):.4f}")

    # You might also find these useful
    print(f"Hamming loss     : {hamming_loss(y_val, preds):.4f}")

    # Example-based metrics (averaged over samples)
    sample_precision = []
    sample_recall = []
    sample_f1 = []

    for i in range(len(y_val)):
        y_true_set = set(np.where(y_val[i])[0])
        y_pred_set = set(np.where(preds[i])[0])

        if len(y_pred_set) == 0:
            p = 1.0 if len(y_true_set) == 0 else 0.0
        else:
            p = len(y_true_set & y_pred_set) / len(y_pred_set)

        if len(y_true_set) == 0:
            r = 1.0 if len(y_pred_set) == 0 else 0.0
        else:
            r = len(y_true_set & y_pred_set) / len(y_true_set)

        if p + r == 0:
            f = 0.0
        else:
            f = 2 * p * r / (p + r)

        sample_precision.append(p)
        sample_recall.append(r)
        sample_f1.append(f)

    print(f"Example‑based precision : {np.mean(sample_precision):.4f}")
    print(f"Example‑based recall    : {np.mean(sample_recall):.4f}")
    print(f"Example‑based F1        : {np.mean(sample_f1):.4f}")


def interactive_predict(clf, vect, thr, labels):
    print("\nEnter text to classify (type 'quit' to exit):")
    while True:
        txt = input("> ")
        if txt.strip().lower() == "quit":
            break
        X = vect.transform([txt])
        sc = clf.decision_function(X)[0]
        pred = (sc >= thr).astype(int)
        emos = [labels[i] for i, flag in enumerate(pred) if flag]
        print("Emotions:", emos or ["(none)"])


def save_model(clf, vect, thr, labels, mlb, model_dir="model"):
    """Save the trained model and associated components."""
    # Create model directory if it doesn't exist
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    # Save all components
    joblib.dump(clf, os.path.join(model_dir, "classifier.joblib"))
    joblib.dump(vect, os.path.join(model_dir, "vectorizer.joblib"))
    joblib.dump(thr, os.path.join(model_dir, "thresholds.joblib"))
    joblib.dump(labels, os.path.join(model_dir, "labels.joblib"))
    joblib.dump(mlb, os.path.join(model_dir, "multilabel_binarizer.joblib"))

    print(f"Model successfully saved to {model_dir}/")


def load_model(model_dir="model"):
    """Load the trained model and associated components."""
    clf = joblib.load(os.path.join(model_dir, "classifier.joblib"))
    vect = joblib.load(os.path.join(model_dir, "vectorizer.joblib"))
    thr = joblib.load(os.path.join(model_dir, "thresholds.joblib"))
    labels = joblib.load(os.path.join(model_dir, "labels.joblib"))
    mlb = joblib.load(os.path.join(model_dir, "multilabel_binarizer.joblib"))

    print(f"Model successfully loaded from {model_dir}/")
    return clf, vect, thr, labels, mlb


def main():
    # Check if model exists and ask if user wants to retrain
    model_dir = "model"
    retrain = True

    if os.path.exists(model_dir) and os.path.isfile(os.path.join(model_dir, "classifier.joblib")):
        response = input("Trained model found. Do you want to retrain? (y/n): ")
        retrain = response.lower() == 'y'

    if retrain:
        print("Training new model...")
        X_train, y_train, X_val, y_val, labels, vect, mlb = load_and_preprocess()
        clf = train_svm(X_train, y_train)
        thr = tune_thresholds(clf, X_val, y_val)
        evaluate(clf, X_val, y_val, thr, labels)

        # Save the model
        save_model(clf, vect, thr, labels, mlb, model_dir)
    else:
        print("Loading existing model...")
        clf, vect, thr, labels, mlb = load_model(model_dir)

    # Interactive prediction
    interactive_predict(clf, vect, thr, labels)


if __name__ == "__main__":
    main()