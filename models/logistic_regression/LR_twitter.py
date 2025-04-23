from datasets import load_dataset
import pandas as pd
import joblib
import os
import zipfile

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report


# This fetches three splits: train, validation, test.
dataset = load_dataset('dair-ai/emotion')

# Convert splits to pandas for easier handling
df_train = pd.DataFrame(dataset['train'])
df_val   = pd.DataFrame(dataset['validation'])
df_test  = pd.DataFrame(dataset['test'])

# Combine train + validation into one larger training set
df_train_full = pd.concat([df_train, df_val], ignore_index=True)

X_train, y_train = df_train_full['text'], df_train_full['label']
X_test,  y_test  = df_test['text'],         df_test['label']

# Vectorize the text data using TF-IDF
vectorizer = TfidfVectorizer(
    stop_words='english',   # remove common English stop words
    max_features=5000,      # only keep the top 5,000 tokens by term frequency
    ngram_range=(1,2),      # uni‑ and bi‑grams
    max_df=0.8,             # ignore terms in >80% of docs
    min_df=5,               # ignore terms in <5 docs
    norm='l2',              # normalize each vector to unit L2 norm
    use_idf=True            # enable inverse-document-frequency reweighting
)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec  = vectorizer.transform(X_test)

# Initialize and train the Logistic Regression model
model = LogisticRegression(
    penalty='l2',            # regularization norm ('l1', 'l2', 'elasticnet', or 'none')
    C=1.0,                   # inverse of regularization strength; smaller → stronger regularization
    solver='lbfgs',          # optimization algorithm: 'newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'
    multi_class='multinomial', # use true multinomial loss for multiclass problems
    max_iter=1000,           # max number of iterations for the solver to converge
    class_weight=None,       # can set 'balanced' or dict to adjust for class imbalance
    random_state=42,         # seed for reproducibility
    tol=1e-4                 # tolerance for stopping criteria
)
model.fit(X_train_vec, y_train)

# Evaluate the model on the test set
label_names = dataset['train'].features['label'].names
y_pred = model.predict(X_test_vec)
print(classification_report(y_test, y_pred, target_names=label_names))


# Interactive prediction function
def predict_emotion(text: str) -> None:
    """
    Transforms the input text with the same TF-IDF vectorizer
    and outputs the predicted emotion label.
    """
    vec        = vectorizer.transform([text])
    pred_label = model.predict(vec)[0]
    print(f"Predicted   : {label_names[pred_label]}")

def export_and_zip(model, vectorizer,
                   export_dir: str = 'export',
                   zip_filename: str = 'model_package.zip') -> None:
    
    os.makedirs(export_dir, exist_ok=True)
    model_path = os.path.join(export_dir, 'emotion_clf.joblib')
    vec_path   = os.path.join(export_dir, 'tfidf_vectorizer.joblib')

    # Serialize to disk
    joblib.dump(model, model_path)
    joblib.dump(vectorizer, vec_path)

    # Create ZIP archive
    with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as z:
        z.write(model_path, arcname='emotion_clf.joblib')
        z.write(vec_path,   arcname='tfidf_vectorizer.joblib')

    print(f"Exported and zipped to {zip_filename}")

#export_and_zip(model, vectorizer)

# Interactive loop for user input
if __name__ == "__main__":
    print("\nEnter a sentence to classify its emotions (type 'quit' to exit):")
    while True:
        sentence = input("> ")
        if sentence.lower() == "quit":
            break
        predict_emotion(sentence)