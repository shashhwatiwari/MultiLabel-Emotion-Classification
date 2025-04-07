# Multi-Label Emotion Classification

This project compares the performance of different NLP models (SVM, Logistic Regression, CNN, BERT, RoBERTa) on multi-label emotion classification tasks. The study focuses on understanding how model performance varies based on the training data distribution.

## Project Structure

```
├── data/
│   ├── raw/                 # Raw datasets
│   ├── processed/           # Processed datasets
│   └── splits/              # Train/val/test splits
├── models/
│   ├── svm/                 # SVM implementation
│   ├── logistic_regression/ # Logistic Regression implementation
│   ├── cnn/                 # CNN implementation
│   ├── bert/                # BERT implementation
│   └── roberta/             # RoBERTa implementation
├── utils/
│   ├── data_processing.py   # Data preprocessing utilities
│   ├── evaluation.py        # Evaluation metrics
│   └── visualization.py     # Visualization utilities
├── experiments/
│   └── results/             # Experiment results and analysis
└── notebooks/
    └── analysis.ipynb       # Analysis and visualization notebooks
```

## Datasets

1. GoEmotions (Training)
   - Multi-label emotion classification dataset
   - 27 emotion categories
   - Reddit comments

2. Movie Reviews (Training)
   - Multi-label sentiment analysis
   - Movie review data

3. Twitter Chats (Testing)
   - Multi-label emotion classification
   - Social media conversations

## Models

- Support Vector Machine (SVM)
- Logistic Regression
- Convolutional Neural Network (CNN)
- BERT
- RoBERTa

## Requirements

- Python 3.8+
- PyTorch
- Transformers
- Scikit-learn
- Pandas
- NumPy
- Matplotlib/Seaborn

## Setup

1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Download datasets
4. Run preprocessing scripts
5. Train and evaluate models

## Evaluation Metrics

- F1 Score (micro and macro)
- Precision
- Recall
- Hamming Loss
- Jaccard Similarity Score

## Results

Results will be stored in the `experiments/results` directory, including:
- Model performance metrics
- Training curves
- Confusion matrices
- Cross-dataset analysis
