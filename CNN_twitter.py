import numpy as np
from datasets import load_dataset
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report

# 1. Load dataset
dataset     = load_dataset("dair-ai/emotion")
train_texts = dataset["train"]["text"]
train_labels= dataset["train"]["label"]
val_texts   = dataset["validation"]["text"]
val_labels  = dataset["validation"]["label"]
test_texts  = dataset["test"]["text"]
test_labels = dataset["test"]["label"]

label_names = dataset["train"].features["label"].names
num_classes = len(label_names)

# 2. Hyperparameters (same as before)
vocab_size     = 10000
embedding_dim  = 100
max_length     = 100
padding_type   = 'post'
trunc_type     = 'post'
oov_token      = "<OOV>"

filters        = 128
kernel_size    = 5
hidden_units   = 64
dropout_rate   = 0.5

learning_rate  = 1e-3
batch_size     = 32
epochs         = 10

# 3. Tokenizer and prep function
tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_token)
tokenizer.fit_on_texts(train_texts)

def prep_texts(texts):
    seqs   = tokenizer.texts_to_sequences(texts)
    return pad_sequences(seqs, maxlen=max_length,
                         padding=padding_type,
                         truncating=trunc_type)

X_train = prep_texts(train_texts)
X_val   = prep_texts(val_texts)
X_test  = prep_texts(test_texts)

y_train = np.eye(num_classes)[train_labels]
y_val   = np.eye(num_classes)[val_labels]
y_test  = np.eye(num_classes)[test_labels]

# 4. Build & compile the 1D‑CNN
model = Sequential([
    Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_length),
    Conv1D(filters=filters, kernel_size=kernel_size, activation='relu'),
    GlobalMaxPooling1D(),
    Dense(hidden_units, activation='relu'),
    Dropout(dropout_rate),
    Dense(num_classes, activation='softmax')
])
model.compile(
    loss='categorical_crossentropy',
    optimizer=Adam(learning_rate=learning_rate),
    metrics=['accuracy']
)
model.summary()

# 5. Train
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    batch_size=batch_size,
    epochs=epochs
)

# 6. Testing & Classification Report
# 6a. Evaluate overall test loss & accuracy
test_loss, test_acc = model.evaluate(X_test, y_test, batch_size=batch_size)
print(f"\nTest Accuracy: {test_acc:.2%} — Loss: {test_loss:.4f}\n")

# 6b. Detailed per-class metrics
y_pred_probs = model.predict(X_test, batch_size=batch_size)
y_pred       = np.argmax(y_pred_probs, axis=1)
y_true       = np.argmax(y_test, axis=1)

report = classification_report(
    y_true,
    y_pred,
    target_names=label_names,
    digits=4
)
print("Classification Report:\n")
print(report)

# 7. Interactive prediction helper (unchanged)
def predict_emotion(text: str):
    seq    = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=max_length,
                           padding=padding_type,
                           truncating=trunc_type)
    probs  = model.predict(padded)[0]
    idx    = np.argmax(probs)
    return label_names[idx], probs[idx]

def interactive_predict():
    print("Enter a sentence (or 'quit' to stop):")
    while True:
        text = input("> ")
        if text.lower() in ('quit', 'exit'):
            break
        label, confidence = predict_emotion(text)
        print(f"Predicted: {label}  (confidence {confidence:.2%})")
interactive_predict()