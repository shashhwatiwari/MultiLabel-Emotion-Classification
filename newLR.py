# 1. Install prerequisites (run once)
# !pip install datasets scikit-learn imbalanced-learn

# 2. Imports
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import classification_report, accuracy_score, f1_score, precision_recall_curve
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import RandomOverSampler
import numpy as np

# 3. Load GoEmotions and extract emotion names
dataset = load_dataset("go_emotions")
emotion_labels = dataset["train"].features["labels"].feature.names

# 4. Prepare texts and raw label lists
train_texts, train_labels = dataset["train"]["text"], dataset["train"]["labels"]
val_texts,   val_labels   = dataset["validation"]["text"], dataset["validation"]["labels"]

# 5. Binarize labels (ensure classes=0…27)
mlb     = MultiLabelBinarizer(classes=list(range(len(emotion_labels))))
y_train = mlb.fit_transform(train_labels)
y_val   = mlb.transform(val_labels)

# 6. Text → TF‑IDF
vectorizer = TfidfVectorizer(
    max_features=10_000,
    ngram_range=(1,2),
    stop_words='english'
)
X_train = vectorizer.fit_transform(train_texts)
X_val   = vectorizer.transform(val_texts)

# 7. Build base LogisticRegression (with class_weight balancing)
base_clf = LogisticRegression(
    penalty='l2',
    C=1.0,
    solver='saga',
    max_iter=1000,
    class_weight='balanced',
    multi_class='ovr'
)

# 8. Wrap in an oversampling pipeline for each one-vs-rest binary task
pipeline = Pipeline([
    ('oversample', RandomOverSampler(random_state=42)),
    ('clf',       base_clf)
])
ovr = OneVsRestClassifier(pipeline, n_jobs=-1)

# 9. Train
ovr.fit(X_train, y_train)

# 10. Get decision scores for threshold tuning
scores = ovr.decision_function(X_val)  # shape: (n_samples, n_classes)

# 11. Tune per-class thresholds by maximizing F1 on validation
best_thresholds = []
for i in range(y_val.shape[1]):
    p, r, thresh = precision_recall_curve(y_val[:, i], scores[:, i])
    f1 = 2 * p * r / (p + r + 1e-8)
    best_thresholds.append(thresh[np.argmax(f1)])

# 12. Predict using tuned thresholds
y_pred_tuned = (scores >= np.array(best_thresholds)).astype(int)

# 13. Detailed per‑label report
print("=== Classification Report (tuned thresholds) ===")
print(classification_report(y_val, y_pred_tuned, target_names=emotion_labels))

# 14. Raw subset accuracy
acc = accuracy_score(y_val, y_pred_tuned)
print(f"Raw Accuracy: {acc:.4f}")

# 15. Micro‑averaged F1
micro_f1 = f1_score(y_val, y_pred_tuned, average="micro")
print(f"Micro-F1:     {micro_f1:.4f}")

# (Optional) Other F1 averages
macro_f1    = f1_score(y_val, y_pred_tuned, average="macro")
weighted_f1 = f1_score(y_val, y_pred_tuned, average="weighted")
print(f"Macro-F1:     {macro_f1:.4f}")
print(f"Weighted-F1:  {weighted_f1:.4f}")

# 16. Inference on custom sentences
def predict_emotions(text):
    # Vectorize
    X = vectorizer.transform([text])
    # Get decision scores
    scores = ovr.decision_function(X)
    # Apply tuned thresholds
    preds = (scores >= np.array(best_thresholds)).astype(int)
    # Return list of emotion labels with a positive prediction
    return [emotion_labels[i] for i, val in enumerate(preds[0]) if val == 1]

if __name__ == "__main__":
    print("\nEnter a sentence to classify its emotions (type 'quit' to exit):")
    while True:
        sentence = input("> ")
        if sentence.lower() == "quit":
            break
        emotions = predict_emotions(sentence)
        print("Predicted emotions:", emotions if emotions else ["(none)"])