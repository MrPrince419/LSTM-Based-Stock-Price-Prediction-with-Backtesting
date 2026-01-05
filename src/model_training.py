"""
Model Training Module
Trains LSTM model with data preprocessing and evaluation.
"""

import numpy as np
import pandas as pd
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import pickle
import os
from typing import Tuple, Optional, Dict
import sys

# Add parent directory to path to import model
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from models.model import LSTMPredictor


class ModelTrainer:
    """
    Handles LSTM model training with data preprocessing.
    """
    
    def __init__(self, lookback_window: int = 60, lstm_units: list = [64, 32],
                 dropout_rate: float = 0.2, learning_rate: float = 0.001):
        """
        Initialize model trainer.
        
        Args:
            lookback_window: Number of days to look back (default: 60)
            lstm_units: List of units for each LSTM layer (default: [64, 32])
            dropout_rate: Dropout rate for regularization (default: 0.2)
            learning_rate: Learning rate for optimizer (default: 0.001)
        """
        self.lookback_window = lookback_window
        self.predictor = LSTMPredictor(lookback_window, lstm_units, dropout_rate, learning_rate)
        self.scaler = StandardScaler()
        self.is_fitted = False
        
    def fit_scaler(self, X_train: np.ndarray) -> None:
        """
        Fit scaler on training data.
        
        Args:
            X_train: Training sequences (samples, timesteps, features)
        """
        # Reshape for scaling: (samples * timesteps, features)
        n_samples, n_timesteps, n_features = X_train.shape
        X_reshaped = X_train.reshape(-1, n_features)
        
        # Fit scaler
        self.scaler.fit(X_reshaped)
        self.is_fitted = True
    
    def transform_data(self, X: np.ndarray) -> np.ndarray:
        """
        Scale input data using fitted scaler.
        
        Args:
            X: Input sequences (samples, timesteps, features)
            
        Returns:
            Scaled sequences
        """
        if not self.is_fitted:
            raise ValueError("Scaler not fitted. Call fit_scaler() first.")
        
        n_samples, n_timesteps, n_features = X.shape
        X_reshaped = X.reshape(-1, n_features)
        X_scaled = self.scaler.transform(X_reshaped)
        X_scaled = X_scaled.reshape(n_samples, n_timesteps, n_features)
        
        return X_scaled
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None,
              epochs: int = 50, batch_size: int = 32, verbose: int = 1) -> dict:
        """
        Train the LSTM model.
        
        Args:
            X_train: Training sequences
            y_train: Training targets
            X_val: Validation sequences (optional)
            y_val: Validation targets (optional)
            epochs: Number of training epochs
            batch_size: Batch size for training
            verbose: Verbosity level
            
        Returns:
            Training history dictionary
        """
        # Fit scaler on training data
        print("Fitting scaler on training data...")
        self.fit_scaler(X_train)
        
        # Scale data
        print("Scaling data...")
        X_train_scaled = self.transform_data(X_train)
        
        # Build model
        input_shape = (X_train_scaled.shape[1], X_train_scaled.shape[2])
        print(f"Building model with input shape: {input_shape}")
        self.predictor.model = self.predictor.build_model(input_shape)
        
        print("\nModel Architecture:")
        self.predictor.model.summary()
        
        # Prepare validation data
        validation_data = None
        if X_val is not None and y_val is not None:
            X_val_scaled = self.transform_data(X_val)
            validation_data = (X_val_scaled, y_val)
        
        # Callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_loss' if validation_data else 'loss',
                patience=10,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss' if validation_data else 'loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7,
                verbose=1
            )
        ]
        
        # Train model
        print("\nTraining model...")
        history = self.predictor.model.fit(
            X_train_scaled,
            y_train,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=verbose
        )
        
        return history.history
    
    def predict(self, X: np.ndarray, return_proba: bool = False) -> np.ndarray:
        """
        Make predictions on new data.
        
        Args:
            X: Input sequences
            return_proba: If True, return probabilities; if False, return binary predictions
            
        Returns:
            Predictions (probabilities or binary)
        """
        X_scaled = self.transform_data(X)
        return self.predictor.predict(X_scaled, return_proba=return_proba)
    
    def predict_with_confidence(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make predictions with confidence scores.
        
        Args:
            X: Input sequences
            
        Returns:
            Tuple of (predictions, confidence_scores)
        """
        X_scaled = self.transform_data(X)
        return self.predictor.predict_with_confidence(X_scaled)
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> dict:
        """
        Evaluate model performance.
        
        Args:
            X_test: Test sequences
            y_test: Test targets
            
        Returns:
            Dictionary with evaluation metrics
        """
        y_pred = self.predict(X_test)
        y_pred_proba = self.predict(X_test, return_proba=True)
        
        accuracy = accuracy_score(y_test, y_pred)
        
        # Classification report
        report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        metrics = {
            'accuracy': accuracy,
            'precision': report['1']['precision'],
            'recall': report['1']['recall'],
            'f1_score': report['1']['f1-score'],
            'confusion_matrix': cm.tolist(),
            'classification_report': report
        }
        
        return metrics
    
    def save_model(self, filepath: str) -> None:
        """
        Save model and scaler to disk.
        
        Args:
            filepath: Base filepath (without extension)
        """
        if self.predictor.model is None:
            raise ValueError("No model to save. Train model first.")
        
        # Save model
        model_path = f"{filepath}_model.h5"
        self.predictor.save_model(model_path)
        
        # Save scaler
        scaler_path = f"{filepath}_scaler.pkl"
        with open(scaler_path, 'wb') as f:
            pickle.dump(self.scaler, f)
        print(f"Scaler saved to {scaler_path}")
    
    def load_model(self, filepath: str) -> None:
        """
        Load model and scaler from disk.
        
        Args:
            filepath: Base filepath (without extension)
        """
        # Load model
        model_path = f"{filepath}_model.h5"
        self.predictor.load_model(model_path)
        
        # Load scaler
        scaler_path = f"{filepath}_scaler.pkl"
        with open(scaler_path, 'rb') as f:
            self.scaler = pickle.load(f)
        self.is_fitted = True
        print(f"Scaler loaded from {scaler_path}")

