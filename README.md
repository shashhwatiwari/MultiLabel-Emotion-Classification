# Emotion Analysis Models

This repository contains implementation of various machine learning and deep learning models for emotion analysis on multiple datasets. The project focuses on comparing different approaches for emotion detection and classification.

## Project Structure

```
├── data_scripts/
│   ├── data_processing.py       # Main data processing utilities
│   ├── data_processing_test.py  # Tests for data processing functions
│   ├── Emotion_DAIR_Analysis.ipynb  # Analysis of DAIR emotion dataset
│   └── GoEmotions_Analysis.ipynb    # Analysis of Google's GoEmotions dataset
├── models/
│   ├── bert/                    # BERT model implementations
│   ├── cnn/
│   │   ├── CNN_geoemotions.ipynb    # CNN model for GoEmotions dataset
│   │   └── CNN_twitter.ipynb        # CNN model for Twitter dataset
│   ├── logistic_regression/
│   │   ├── LR_geoemotions.py        # Logistic Regression for GoEmotions
│   │   └── LR_twitter.py            # Logistic Regression for Twitter
│   ├── roberta/
│   │   ├── RoBERTA_GoEmotions.ipynb # RoBERTa model for GoEmotions
│   │   └── roBERTa_Twitter-2.ipynb  # RoBERTa model for Twitter (v2)
│   └── svm/
│       ├── svm_geomotion.py         # SVM for GoEmotions
│       └── svm_twitter.py           # SVM for Twitter
├── model_test.py                # Model evaluation and testing script
├── .gitignore                   # Git ignore file
├── README.md                    # This file
└── requirements.txt             # Project dependencies
```

## Datasets

This project works with multiple emotion datasets:

- **GoEmotions**: Google's dataset containing 58k Reddit comments labeled with 27 emotions
- **Twitter**: Emotion-labeled tweets dataset
- **DAIR**: Additional emotion analysis dataset

## Models Implemented

The repository contains implementations of various models for emotion classification:

- **CNN**: Convolutional Neural Networks for text classification
- **BERT**: Bidirectional Encoder Representations from Transformers
- **RoBERTa**: Robustly Optimized BERT Pretraining Approach
- **SVM**: Support Vector Machines
- **Logistic Regression**: Traditional machine learning approach

## Getting Started

1. Clone the repository
2. Install requirements:
   ```
   pip install -r requirements.txt
   ```
3. Run data processing scripts to prepare datasets:
   ```
   python data_scripts/data_processing.py
   ```
4. Train models using either Python scripts or Jupyter notebooks in the respective model directories

## Usage

Each model directory contains either Python scripts or Jupyter notebooks that can be used to train and evaluate models on different datasets.

For example, to train a CNN model on the GoEmotions dataset:
1. Open `models/cnn/CNN_geoemotions.ipynb` in Jupyter
2. Follow the instructions in the notebook to load data, train the model, and evaluate results

## Requirements

See `requirements.txt` for a full list of dependencies.

## Contributors

Sanshrit Bakshi
Shashwat Tiwari
Sanidhya Maharia