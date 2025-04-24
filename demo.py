import numpy as np
import torch
import joblib
import pickle
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
)
from huggingface_hub import hf_hub_download
from sklearn.metrics import classification_report, f1_score, hamming_loss
import matplotlib.pyplot as plt
import gradio as gr

import sys
import keras_preprocessing
import sys
import importlib

# import the real keras-preprocessing submodules
kp_root = importlib.import_module("keras_preprocessing")
kp_seq  = importlib.import_module("keras_preprocessing.sequence")
kp_txt  = importlib.import_module("keras_preprocessing.text")

# alias them into keras.src.preprocessing.* so pickle finds them
sys.modules["keras.src.preprocessing"]          = kp_root
sys.modules["keras.src.preprocessing.sequence"] = kp_seq
sys.modules["keras.src.preprocessing.text"]     = kp_txt

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load test split & label names
test_ds    = load_dataset("go_emotions", split="test")
label_names = test_ds.features["labels"].feature.names
N_CLASSES   = len(label_names)

# Model registry: add your HF repos here
MODEL_REGISTRY = {
    # — Transformers —
    "DistilBERT": {
        "type":    "transformer",
        "repo_id": "BakshiSan/distilbert_GoEmo"
    },
    "RoBERTa": {
        "type":    "transformer",
        "repo_id": "BakshiSan/RoBERTa_GoEmotions"
    },
    # — scikit‑learn pipeline —
    "LogisticRegression": {
        "type":    "sklearn",
        "repo_id": "BakshiSan/LR_GoEmotions",
        "files":   [
            "vectorizer.joblib",
            "classifier.joblib",
            "multilabel_binarizer.joblib",
            "thresholds.joblib",
            "emotion_labels.joblib"
        ]
    },
        "SVM": {
        "type":    "sklearn",
        "repo_id": "BakshiSan/SVM_GoEmotions",
        "files":   [
            "vectorizer.joblib",
            "classifier.joblib",
            "multilabel_binarizer.joblib",
            "thresholds.joblib",
            "emotion_labels.joblib"
        ]
    },
    "BERT":{
        "type": "transformer",
        "repo_id": "BakshiSan/BERT"
    },
        "BERT_ClassBalanced":{
        "type": "transformer",
        "repo_id": "BakshiSan/BERT_ClassBalanced"
    },
        # — TensorFlow‑Keras CNN —
    "CNN": {
        "type":           "keras_cnn",
        "repo_id":        "BakshiSan/CNN_GoEmotions",
        "model_file":     "cnn_goemotions.h5",
        "tokenizer_file": "tokenizer_goemotions.pkl",
        "max_length":     100
    }
}

# Load all models and tokenizers into memory
loaded = {}

for name, info in MODEL_REGISTRY.items():
    if info["type"] == "transformer":
        tok = AutoTokenizer.from_pretrained(info["repo_id"])
        mdl = AutoModelForSequenceClassification.from_pretrained(info["repo_id"]).to(device).eval()
        loaded[name] = {"tokenizer": tok, "model": mdl}

    elif info["type"] == "sklearn":
        # download all required files
        paths = {
            fname: hf_hub_download(info["repo_id"], fname)
            for fname in info["files"]
        }
        vectorizer = joblib.load(paths["vectorizer.joblib"])
        classifier = joblib.load(paths["classifier.joblib"])
        mlb        = joblib.load(paths["multilabel_binarizer.joblib"])
        thresholds = joblib.load(paths["thresholds.joblib"])
        labels     = joblib.load(paths["emotion_labels.joblib"])
        loaded[name] = {
            "vectorizer": vectorizer,
            "classifier": classifier,
            "mlb":        mlb,
            "thresholds": thresholds,
            "label_names": labels
        }

    elif info["type"] == "keras_cnn":
        mpath = hf_hub_download(info["repo_id"], info["model_file"])
        tpath = hf_hub_download(info["repo_id"], info["tokenizer_file"])
        cnn_model     = tf.keras.models.load_model(mpath)
        with open(tpath, "rb") as f:
            cnn_tokenizer = pickle.load(f)
        loaded[name] = {
            "model":      cnn_model,
            "tokenizer":  cnn_tokenizer,
            "max_length": info["max_length"]
        }

def evaluate(text, model_name):
    try:
        entry = loaded[model_name]

        # Set threshold based on model
        model_type = MODEL_REGISTRY[model_name]["type"]
        if model_type == "transformer":
            threshold = 0.3
        elif model_type == "sklearn":
            threshold = 0.6
        elif model_type == "keras_cnn":
            threshold = 0.3
        else:
            threshold = 0.3  # default

        # — Transformer branch —
        if MODEL_REGISTRY[model_name]["type"] == "transformer":
            tok, mdl = entry["tokenizer"], entry["model"]
            enc = tok(
                text,
                return_tensors="pt",
                truncation=True,
                padding="max_length",
                max_length=128
            ).to(device)
            with torch.no_grad():
                probs = torch.sigmoid(mdl(**enc).logits)[0].cpu().numpy()

        # — Sklearn branch —
        elif MODEL_REGISTRY[model_name]["type"] == "sklearn":
            vect = entry["vectorizer"]
            clf  = entry["classifier"]
            X    = vect.transform([text])
            if hasattr(clf, "predict_proba"):
                probs = clf.predict_proba(X)[0]
            else:
                scores = clf.decision_function(X)[0]
                probs  = 1 / (1 + np.exp(-scores))

        # — Keras‑CNN branch —
        elif MODEL_REGISTRY[model_name]["type"] == "keras_cnn":
            cnn_tok = entry["tokenizer"]
            cnn_mdl = entry["model"]
            maxlen  = entry["max_length"]
            seq = cnn_tok.texts_to_sequences([text])
            pad = pad_sequences(seq, maxlen=maxlen, padding="post", truncating="post")
            probs = cnn_mdl.predict(pad, verbose=0)[0]

        else:
            raise ValueError(f"Unknown model type for {model_name}")

        pred_vec = (probs > threshold).astype(int)  # shape (N_CLASSES,)
        labels   = [label_names[i] for i, flag in enumerate(pred_vec) if flag == 1]

        return ", ".join(labels)

    except Exception:
        tb = traceback.format_exc()
        return f"❌ Error:\n{tb}", plt.figure()

# Gradio demo
import traceback

def predict_fn(text, model_name):
    try:
        entry = loaded[model_name]

        # Set threshold based on model type
        model_type = MODEL_REGISTRY[model_name]["type"]
        if model_type == "transformer":
            threshold = 0.3
        elif model_type == "sklearn":
            threshold = 0.6
        elif model_type == "keras_cnn":
            threshold = 0.3
        else:
            threshold = 0.3  # default

        # — Transformer branch —
        if MODEL_REGISTRY[model_name]["type"] == "transformer":
            tok, mdl = entry["tokenizer"], entry["model"]
            enc = tok(
                text,
                return_tensors="pt",
                truncation=True,
                padding="max_length",
                max_length=128
            ).to(device)
            with torch.no_grad():
                probs = torch.sigmoid(mdl(**enc).logits)[0].cpu().numpy()

        # — Sklearn branch: JUST raw probs, no inverse_transform here —
        elif MODEL_REGISTRY[model_name]["type"] == "sklearn":
            vect = entry["vectorizer"]
            clf  = entry["classifier"]
            X    = vect.transform([text])
            if hasattr(clf, "predict_proba"):
                probs = clf.predict_proba(X)[0]
            else:
                scores = clf.decision_function(X)[0]
                probs  = 1 / (1 + np.exp(-scores))

        # — Keras‑CNN branch —
        elif MODEL_REGISTRY[model_name]["type"] == "keras_cnn":
            cnn_tok = entry["tokenizer"]
            cnn_mdl = entry["model"]
            maxlen  = entry["max_length"]
            seq = cnn_tok.texts_to_sequences([text])
            pad = pad_sequences(seq, maxlen=maxlen, padding="post", truncating="post")
            probs = cnn_mdl.predict(pad, verbose=0)[0]

        else:
            raise ValueError(f"Unknown model type for {model_name}")

        # — now ONE consistent threshold→labels step for ALL models —
        pred_vec = (probs > threshold).astype(int)  # shape (N_CLASSES,)
        labels   = [label_names[i] for i, flag in enumerate(pred_vec) if flag == 1]

        # build the bar chart
        fig, ax = plt.subplots(figsize=(6,8))
        ax.barh(label_names, probs)
        ax.set_xlim(0, 1)
        ax.invert_yaxis()
        ax.set_xlabel("Probability")
        # ax.axvline(x=threshold, color='r', linestyle='--', alpha=0.7, label=f'Threshold: {threshold}')
        ax.legend()
        plt.tight_layout()

        return ", ".join(labels), fig

    except Exception:
        tb = traceback.format_exc()
        return f"❌ Error:\n{tb}", plt.figure()

iface = gr.Interface(
    fn=predict_fn,
    inputs=[
        gr.Textbox(lines=3, label="Input Text"),
        gr.Radio(list(MODEL_REGISTRY.keys()), label="Model")
    ],
    outputs=[gr.Textbox(label="Emotions"), gr.Plot()],
    title="Universal Emotion Detection Demo"
)

# Predefined test inputs
test_inputs = [
    "I love this product! It's amazing.",
    "I'm so sad and disappointed."
]
# Run the demo
for model_name in MODEL_REGISTRY.keys():
    print(f"Testing model: {model_name} ===============================================================")
    for text in test_inputs:
        labels = evaluate(text, model_name)
        print(f"Input: {text}\nPredicted Emotions: {labels}\n")

iface.launch()