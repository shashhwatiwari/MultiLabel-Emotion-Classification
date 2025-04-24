# Emotion Classification

This repository contains code and notebooks for analyzing emotions in text data using various machine learning models.

## Project Overview

This project aims to detect and analyze emotions in text data from different sources (Twitter and GoEmotions dataset) using multiple machine learning approaches. The models range from traditional machine learning techniques like SVM and Logistic Regression to more advanced deep learning models such as BERT, DistilBERT, RoBERTa, and CNN.

## Repository Structure

```
.
├── data_scripts/
│   ├── __pycache__/
│   ├── data_processing_test.py
│   ├── data_processing.py
│   ├── Emotion_DAIR_Analysis.ipynb
│   └── GoEmotions_Analysis.ipynb
├── models/
│   ├── bert/
│   │   ├── BERT_Twitter.ipynb
│   │   └── CS6120_BERT_GoEmotions.ipynb
│   ├── cnn/
│   │   ├── CNN_goemotions.ipynb
│   │   └── CNN_twitter.ipynb
│   ├── distilbert/
│   │   ├── DistilBERT_FINAL_GoEmotions.ipynb
│   │   └── DistilBERT_Twitter.ipynb
│   ├── logistic_regression/
│   │   ├── LR_goemotions.py
│   │   └── LR_twitter.py
│   ├── roberta/
│   │   ├── RoBERTA_GoEmotions.ipynb
│   │   └── roBERTa_Twitter-2.ipynb
│   └── svm/
│       ├── svm_goemotion.py
│       ├── svm_twitter.py
│       └── model_test.py
├── .gitignore
├── demo.py
├── README.md
└── requirements.txt
```

## Models

The project implements and compares the following models:

1. **Traditional Machine Learning**
   - Support Vector Machines (SVM)
   - Logistic Regression (LR)

2. **Transformer-based Models**
   - BERT
   - DistilBERT (a lighter version of BERT)
   - RoBERTa

3. **Convolutional Neural Networks (CNN)**

Each model is implemented for both Twitter data and the GoEmotions dataset to compare performance across different data sources.

## Datasets

The project works with two main datasets:
- **Twitter data**: Tweets labeled with emotions
- **GoEmotions**: A dataset of comments from Reddit, labeled with emotions

## Data Processing

The `data_scripts` directory contains scripts for:
- Loading and preprocessing text data
- Feature extraction
- Data transformation for different model architectures
- Analysis of emotion distributions in datasets

## Usage

### Prerequisites

To install the required dependencies:

```bash
pip install -r requirements.txt
```

### Running the Models

- **Jupyter Notebooks**: Open and run the respective `.ipynb` files in the model directories
- **Python Scripts**: Run the `.py` files for the corresponding models

Example:
```bash
python models/svm/svm_twitter.py
```

### Demo

A demonstration script is available:

```bash
python demo.py
```

This allows for quick testing of emotion detection on sample text inputs.

## Contributors

Sanshrit Bakshi
Shashwat Tiwari
Sanidhya Maharia