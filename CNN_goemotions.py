import numpy as np
from datasets import load_dataset
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Embedding, Conv1D, GlobalMaxPooling1D,
    concatenate, Dropout, Dense
)
from tensorflow.keras.optimizers import Adam

# 1. Load GoEmotions
dataset = load_dataset("go_emotions")
train_texts = dataset["train"]["text"]
val_texts   = dataset["validation"]["text"]
test_texts  = dataset["test"]["text"]
train_labels = dataset["train"]["labels"]
val_labels   = dataset["validation"]["labels"]
test_labels  = dataset["test"]["labels"]
label_names  = dataset["train"].features["labels"].feature.names  # 28 emotion labels

# 2. Multi‑hot encode labels
num_labels = len(label_names)
def to_multi_hot(label_lists):
    m = np.zeros((len(label_lists), num_labels), dtype=np.int32)
    for i, labs in enumerate(label_lists):
        m[i, labs] = 1
    return m

y_train = to_multi_hot(train_labels)
y_val   = to_multi_hot(val_labels)
y_test  = to_multi_hot(test_labels)

# 3. Tokenize & pad
vocab_size = 20000  
max_len    = 100     

tokenizer = Tokenizer(num_words=vocab_size, oov_token="<OOV>")
tokenizer.fit_on_texts(train_texts)

def encode(texts):
    seq = tokenizer.texts_to_sequences(texts)
    return pad_sequences(seq, maxlen=max_len, padding="post", truncating="post")

X_train = encode(train_texts)
X_val   = encode(val_texts)
X_test  = encode(test_texts)

# 4. Build the CNN model
embedding_dim = 128       
filter_sizes  = [3,4,5]   
num_filters   = 128       
drop_rate     = 0.5       

inputs = Input(shape=(max_len,), dtype="int32")
embed  = Embedding(vocab_size, embedding_dim, input_length=max_len)(inputs)

conv_blocks = []
for sz in filter_sizes:
    conv = Conv1D(filters=num_filters, kernel_size=sz, activation="relu")(embed)
    pool = GlobalMaxPooling1D()(conv)
    conv_blocks.append(pool)

concat = concatenate(conv_blocks)
drop   = Dropout(drop_rate)(concat)
output = Dense(num_labels, activation="sigmoid")(drop)

model = Model(inputs, output)

# 5. Compile & train
learning_rate = 1e-3
batch_size    = 64
epochs        = 5

model.compile(
    optimizer=Adam(learning_rate=learning_rate),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    batch_size=batch_size,
    epochs=epochs
)

# 6. Evaluate
loss, acc = model.evaluate(X_test, y_test, batch_size=batch_size)
print(f"Test Loss: {loss:.4f}, Test Accuracy: {acc:.4f}")

# 7. Interactive prediction function
def predict_emotions(text, top_k=5, threshold=0.2):
    """
    Predicts the top_k emotions for a given text with scores above threshold.
    Returns a list of (label, score) tuples sorted by score descending.
    """
    seq    = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=max_len, padding='post', truncating='post')
    preds  = model.predict(padded)[0]
    # pair labels with scores and sort
    pairs = sorted(zip(label_names, preds), key=lambda x: x[1], reverse=True)
    return [(label, float(score)) for label, score in pairs if score >= threshold][:top_k]

if __name__ == "__main__":
    print("\nEnter a sentence to classify emotions (type 'quit' to exit):")
    while True:
        user_input = input("> ")
        if user_input.lower() in ("quit", "exit"):
            break
        results = predict_emotions(user_input)
        if results:
            print("Predicted emotions:")
            for label, score in results:
                print(f"  {label:>10s}: {score:.3f}")
        else:
            print("No emotion score exceeded the threshold. Try a different sentence.")