"""
Strategy Module
Generates buy/sell signals based on LSTM predictions.
"""

import numpy as np
from typing import Optional


class TradingStrategy:
    """
    Trend-following trading strategy based on LSTM predictions.
    
    Strategy Rules:
    - BUY when prediction is UP and confidence > buy_threshold (default: 0.5)
      - Uses lenient threshold (min 0.4) since UP predictions tend to have lower confidence
      - Fallback: If no BUY signals, allows UP predictions with confidence > 0.3
    - SELL when prediction is DOWN and confidence < sell_threshold (default: 0.5)
    - HOLD otherwise
    """
    
    def __init__(self, buy_threshold: float = 0.6, sell_threshold: float = 0.4,
                 commission: float = 0.001, initial_capital: float = 100000):
        """
        Initialize trading strategy.
        
        Args:
            buy_threshold: Confidence threshold for BUY signal (default: 0.6)
            sell_threshold: Confidence threshold for SELL signal (default: 0.4)
            commission: Commission rate per trade (default: 0.001 = 0.1%)
            initial_capital: Starting capital (default: 100000)
        """
        self.buy_threshold = buy_threshold
        self.sell_threshold = sell_threshold
        self.commission = commission
        self.initial_capital = initial_capital
        
    def generate_signals(self, predictions: np.ndarray, 
                        confidence: np.ndarray) -> np.ndarray:
        """
        Generate trading signals based on predictions and confidence.
        
        Args:
            predictions: Binary predictions (0 or 1)
            confidence: Confidence scores (0 to 1)
            
        Returns:
            Array of signals: 1 for BUY, -1 for SELL, 0 for HOLD
        """
        signals = np.zeros(len(predictions))
        
        # BUY signal: prediction is UP and confidence > buy_threshold
        # Very lenient for BUY: use minimum threshold of 0.4 since UP predictions tend to have lower confidence
        buy_confidence_threshold = min(self.buy_threshold, 0.4)  # Use the lower of the two
        buy_mask = (predictions == 1) & (confidence > buy_confidence_threshold)
        signals[buy_mask] = 1
        
        # SELL signal: prediction is DOWN and confidence < sell_threshold
        sell_mask = (predictions == 0) & (confidence < self.sell_threshold)
        signals[sell_mask] = -1
        
        # Fallback: If still no BUY signals, be even more lenient
        if sum(signals == 1) == 0 and sum(predictions == 1) > 0:
            # Allow any UP prediction with confidence > 0.3
            buy_mask_fallback = (predictions == 1) & (confidence > 0.3)
            signals[buy_mask_fallback] = 1
        
        return signals

