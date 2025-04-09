# Emotion Classification Models

This directory contains code for emotion classification using BERT models.

## Model Files

The trained models are not included in the repository due to their large size. To use the code, you need to:

1. Create the following directory structure:
```
models/bert/saved_models/
    └── go_emotions/
        ├── go_emotions_model/
        └── go_emotions_tokenizer/
```

2. Place the model files in the appropriate directories:
- Unzip `go_emotions_model.zip` into `go_emotions_model/`
- Unzip `go_emotions_tokenizer.zip` into `go_emotions_tokenizer/`

## Usage

1. Training:
```bash
python train_goemotions.py
```

2. Prediction:
```bash
python predict.py
```

## Model Sources

- GoEmotions model: Trained on the GoEmotions dataset
- (Future) Twitter model: Will be trained on Twitter emotion dataset

## Results

Results from model evaluation will be saved in:
```
models/bert/results/
``` 