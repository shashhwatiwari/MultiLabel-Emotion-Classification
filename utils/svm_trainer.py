import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import os
from typing import Tuple, List, Dict, Union
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SVMTrainer:
    def __init__(
        self,
        ngram_range: Tuple[int, int] = (1, 2),
        max_df: float = 0.9,
        min_df: int = 5,
        C: float = 1.0,
        max_iter: int = 10000,
        is_multilabel: bool = True
    ):
        """
        Initialize SVM trainer with TF-IDF vectorizer and SVM classifier.
        
        Args:
            ngram_range: Range of n-grams to consider
            max_df: Ignore terms that appear in more than this proportion of documents
            min_df: Ignore terms that appear in fewer than this number of documents
            C: Regularization parameter for SVM
            max_iter: Maximum number of iterations for SVM
            is_multilabel: Whether to use multilabel classification
        """
        self.ngram_range = ngram_range
        self.max_df = max_df
        self.min_df = min_df
        self.C = C
        self.max_iter = max_iter
        self.is_multilabel = is_multilabel
        
        # Initialize vectorizer
        self.vectorizer = TfidfVectorizer(
            preprocessor=self._preprocess_text,
            tokenizer=None,
            ngram_range=ngram_range,
            max_df=max_df,
            min_df=min_df
        )
        
        # Initialize classifier
        base_clf = LinearSVC(
            C=C,
            class_weight='balanced',
            max_iter=max_iter
        )
        
        if is_multilabel:
            self.classifier = OneVsRestClassifier(base_clf)
        else:
            self.classifier = base_clf
        
        # Create pipeline
        self.pipeline = Pipeline([
            ('vectorizer', self.vectorizer),
            ('classifier', self.classifier)
        ])
        
        # Initialize label encoder for single-label classification
        if not is_multilabel:
            self.label_encoder = LabelEncoder()
    
    def _preprocess_text(self, text: str) -> str:
        """
        Basic text preprocessing.
        Can be extended with more sophisticated preprocessing.
        """
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = ' '.join(word for word in text.split() if not word.startswith('http'))
        
        # Remove special characters and numbers
        text = ''.join(c for c in text if c.isalpha() or c.isspace())
        
        return text
    
    def prepare_data(
        self,
        texts: List[str],
        labels: Union[List[str], np.ndarray],
        test_size: float = 0.2,
        val_size: float = 0.1,
        random_state: int = 42
    ) -> Tuple:
        """
        Split data into train, validation, and test sets.
        
        Args:
            texts: List of text documents
            labels: List of labels or numpy array of one-hot encoded labels
            test_size: Proportion of data to use for testing
            val_size: Proportion of training data to use for validation
            random_state: Random seed for reproducibility
            
        Returns:
            Tuple of (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        # Initial split into train and test
        X_train, X_test, y_train, y_test = train_test_split(
            texts, labels,
            test_size=test_size,
            random_state=random_state,
            stratify=labels if not self.is_multilabel else None
        )
        
        # Split train into train and validation
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train,
            test_size=val_size,
            random_state=random_state,
            stratify=y_train if not self.is_multilabel else None
        )
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def tune_hyperparameters(
        self,
        X_train: List[str],
        y_train: Union[List[str], np.ndarray],
        param_grid: Dict = None,
        cv: int = 5,
        scoring: str = 'f1_macro',
        n_jobs: int = -1
    ) -> Dict:
        """
        Perform grid search for hyperparameter tuning.
        
        Args:
            X_train: Training texts
            y_train: Training labels
            param_grid: Dictionary of parameters to search
            cv: Number of cross-validation folds
            scoring: Scoring metric
            n_jobs: Number of jobs to run in parallel
            
        Returns:
            Dictionary of best parameters
        """
        if param_grid is None:
            param_grid = {
                'vectorizer__ngram_range': [(1, 1), (1, 2)],
                'vectorizer__min_df': [1, 5, 10],
                'classifier__estimator__C': [0.1, 1, 10] if self.is_multilabel else ['classifier__C', [0.1, 1, 10]]
            }
        
        grid = GridSearchCV(
            self.pipeline,
            param_grid,
            cv=cv,
            scoring=scoring,
            n_jobs=n_jobs
        )
        
        logger.info("Starting hyperparameter tuning...")
        grid.fit(X_train, y_train)
        
        logger.info(f"Best parameters: {grid.best_params_}")
        logger.info(f"Best score: {grid.best_score_:.4f}")
        
        # Update pipeline with best parameters
        self.pipeline = grid.best_estimator_
        
        return grid.best_params_
    
    def train(
        self,
        X_train: List[str],
        y_train: Union[List[str], np.ndarray],
        X_val: List[str] = None,
        y_val: Union[List[str], np.ndarray] = None
    ) -> None:
        """
        Train the SVM model.
        
        Args:
            X_train: Training texts
            y_train: Training labels
            X_val: Validation texts (optional)
            y_val: Validation labels (optional)
        """
        logger.info("Starting model training...")
        
        # Fit the pipeline
        self.pipeline.fit(X_train, y_train)
        
        # Evaluate on validation set if provided
        if X_val is not None and y_val is not None:
            y_pred = self.pipeline.predict(X_val)
            logger.info("\nValidation Set Performance:")
            logger.info(classification_report(y_val, y_pred))
    
    def evaluate(
        self,
        X_test: List[str],
        y_test: Union[List[str], np.ndarray],
        target_names: List[str] = None
    ) -> Dict:
        """
        Evaluate the model on test set.
        
        Args:
            X_test: Test texts
            y_test: Test labels
            target_names: Names of target classes
            
        Returns:
            Dictionary containing evaluation metrics
        """
        logger.info("Evaluating model on test set...")
        
        # Make predictions
        y_pred = self.pipeline.predict(X_test)
        
        # Calculate metrics
        report = classification_report(
            y_test, y_pred,
            target_names=target_names,
            output_dict=True
        )
        
        # Calculate confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        logger.info("\nTest Set Performance:")
        logger.info(classification_report(y_test, y_pred, target_names=target_names))
        
        return {
            'classification_report': report,
            'confusion_matrix': cm
        }
    
    def save_model(self, model_dir: str) -> None:
        """
        Save the trained model and vectorizer.
        
        Args:
            model_dir: Directory to save the model
        """
        os.makedirs(model_dir, exist_ok=True)
        
        # Save pipeline
        joblib.dump(self.pipeline, os.path.join(model_dir, 'svm_pipeline.joblib'))
        
        # Save label encoder if single-label
        if not self.is_multilabel:
            joblib.dump(self.label_encoder, os.path.join(model_dir, 'label_encoder.joblib'))
        
        logger.info(f"Model saved to {model_dir}")
    
    @classmethod
    def load_model(cls, model_dir: str) -> 'SVMTrainer':
        """
        Load a trained model.
        
        Args:
            model_dir: Directory containing the saved model
            
        Returns:
            Loaded SVMTrainer instance
        """
        # Load pipeline
        pipeline = joblib.load(os.path.join(model_dir, 'svm_pipeline.joblib'))
        
        # Create instance
        instance = cls()
        instance.pipeline = pipeline
        
        # Load label encoder if single-label
        if os.path.exists(os.path.join(model_dir, 'label_encoder.joblib')):
            instance.label_encoder = joblib.load(os.path.join(model_dir, 'label_encoder.joblib'))
            instance.is_multilabel = False
        
        logger.info(f"Model loaded from {model_dir}")
        return instance 