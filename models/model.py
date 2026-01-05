"""
LSTM Model Architecture
Defines the LSTM neural network architecture for stock price direction prediction.
"""

import numpy as np
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from typing import Tuple


class LSTMPredictor:
    """
    LSTM-based stock price direction predictor.
    
    Architecture:
    - Input: Sequences of 60 days with multiple features
    - LSTM Layer 1: 64 units
    - LSTM Layer 2: 32 units
    - Dense Output: 1 unit (sigmoid activation for binary classification)
    """
    
    def __init__(self, lookback_window: int = 60, lstm_units: list = [64, 32],
                 dropout_rate: float = 0.2, learning_rate: float = 0.001):
        """
        Initialize LSTM predictor.
        
        Args:
            lookback_window: Number of days to look back (default: 60)
            lstm_units: List of units for each LSTM layer (default: [64, 32])
            dropout_rate: Dropout rate for regularization (default: 0.2)
            learning_rate: Learning rate for optimizer (default: 0.001)
        """
        self.lookback_window = lookback_window
        self.lstm_units = lstm_units
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.model = None
        
    def build_model(self, input_shape: Tuple[int, int]) -> Sequential:
        """
        Build LSTM model architecture.
        
        Args:
            input_shape: Shape of input data (timesteps, features)
            
        Returns:
            Compiled Keras model
        """
        model = Sequential()
        
        # First LSTM layer
        model.add(LSTM(
            units=self.lstm_units[0],
            return_sequences=True,
            input_shape=input_shape
        ))
        model.add(Dropout(self.dropout_rate))
        
        # Second LSTM layer (if specified)
        if len(self.lstm_units) > 1:
            model.add(LSTM(
                units=self.lstm_units[1],
                return_sequences=False
            ))
            model.add(Dropout(self.dropout_rate))
        
        # Output layer
        model.add(Dense(1, activation='sigmoid'))
        
        # Compile model
        model.compile(
            optimizer=Adam(learning_rate=self.learning_rate),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def predict(self, X: np.ndarray, return_proba: bool = False) -> np.ndarray:
        """
        Make predictions on new data.
        
        Args:
            X: Input sequences (already scaled)
            return_proba: If True, return probabilities; if False, return binary predictions
            
        Returns:
            Predictions (probabilities or binary)
        """
        if self.model is None:
            raise ValueError("Model not built. Call build_model() first.")
        
        predictions = self.model.predict(X, verbose=0)
        
        if return_proba:
            return predictions.flatten()
        else:
            return (predictions.flatten() > 0.5).astype(int)
    
    def predict_with_confidence(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make predictions with confidence scores.
        
        Args:
            X: Input sequences (already scaled)
            
        Returns:
            Tuple of (predictions, confidence_scores)
            - predictions: Binary predictions (0 or 1)
            - confidence_scores: Probability scores (0 to 1)
        """
        probabilities = self.predict(X, return_proba=True)
        predictions = (probabilities > 0.5).astype(int)
        
        # Confidence is distance from 0.5
        confidence = np.abs(probabilities - 0.5) * 2
        
        return predictions, confidence
    
    def save_model(self, filepath: str) -> None:
        """
        Save model to disk.
        
        Args:
            filepath: Filepath (with .h5 extension)
        """
        if self.model is None:
            raise ValueError("No model to save. Build model first.")
        
        self.model.save(filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str) -> None:
        """
        Load model from disk.
        
        Args:
            filepath: Filepath (with .h5 extension)
        """
        self.model = keras.models.load_model(filepath)
        print(f"Model loaded from {filepath}")

