"""
Feature Engineering Module
Creates technical indicators and prepares features for LSTM model.
"""

import pandas as pd
import numpy as np
from typing import Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


class FeatureEngineer:
    """
    Engineers technical indicators and prepares features for machine learning.
    
    Technical Indicators:
    - RSI (Relative Strength Index)
    - MACD (Moving Average Convergence Divergence)
    - Moving Averages (SMA, EMA)
    - Bollinger Bands
    - Price changes and returns
    """
    
    def __init__(self, lookback_window: int = 60):
        """
        Initialize feature engineer.
        
        Args:
            lookback_window: Number of days to look back for LSTM (default: 60)
        """
        self.lookback_window = lookback_window
        
    def calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """
        Calculate Relative Strength Index (RSI).
        
        Args:
            prices: Series of closing prices
            period: RSI period (default: 14)
            
        Returns:
            Series with RSI values
        """
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def calculate_macd(self, prices: pd.Series, fast: int = 12, 
                      slow: int = 26, signal: int = 9) -> pd.DataFrame:
        """
        Calculate MACD (Moving Average Convergence Divergence).
        
        Args:
            prices: Series of closing prices
            fast: Fast EMA period (default: 12)
            slow: Slow EMA period (default: 26)
            signal: Signal line period (default: 9)
            
        Returns:
            DataFrame with MACD, Signal, and Histogram columns
        """
        ema_fast = prices.ewm(span=fast, adjust=False).mean()
        ema_slow = prices.ewm(span=slow, adjust=False).mean()
        
        macd = ema_fast - ema_slow
        signal_line = macd.ewm(span=signal, adjust=False).mean()
        histogram = macd - signal_line
        
        return pd.DataFrame({
            'MACD': macd,
            'MACD_Signal': signal_line,
            'MACD_Histogram': histogram
        })
    
    def calculate_bollinger_bands(self, prices: pd.Series, period: int = 20, 
                                 std_dev: float = 2.0) -> pd.DataFrame:
        """
        Calculate Bollinger Bands.
        
        Args:
            prices: Series of closing prices
            period: Moving average period (default: 20)
            std_dev: Number of standard deviations (default: 2.0)
            
        Returns:
            DataFrame with Upper, Middle, and Lower bands
        """
        sma = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()
        
        upper_band = sma + (std * std_dev)
        lower_band = sma - (std * std_dev)
        
        return pd.DataFrame({
            'BB_Upper': upper_band,
            'BB_Middle': sma,
            'BB_Lower': lower_band,
            'BB_Width': upper_band - lower_band,
            'BB_Position': (prices - lower_band) / (upper_band - lower_band)
        })
    
    def calculate_moving_averages(self, prices: pd.Series, 
                                  periods: list = [5, 10, 20, 50]) -> pd.DataFrame:
        """
        Calculate Simple and Exponential Moving Averages.
        
        Args:
            prices: Series of closing prices
            periods: List of periods for moving averages
            
        Returns:
            DataFrame with SMA and EMA columns
        """
        ma_data = {}
        
        for period in periods:
            ma_data[f'SMA_{period}'] = prices.rolling(window=period).mean()
            ma_data[f'EMA_{period}'] = prices.ewm(span=period, adjust=False).mean()
        
        return pd.DataFrame(ma_data)
    
    def calculate_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate price-based features (returns, changes, etc.).
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with price features
        """
        features = pd.DataFrame(index=df.index)
        
        # Returns
        features['Returns'] = df['Close'].pct_change()
        features['Log_Returns'] = np.log(df['Close'] / df['Close'].shift(1))
        
        # Price changes
        features['Price_Change'] = df['Close'].diff()
        features['High_Low_Ratio'] = df['High'] / df['Low']
        features['Close_Open_Ratio'] = df['Close'] / df['Open']
        
        # Volume features
        features['Volume_Change'] = df['Volume'].pct_change()
        features['Volume_MA'] = df['Volume'].rolling(window=20).mean()
        features['Volume_Ratio'] = df['Volume'] / features['Volume_MA']
        
        # Volatility
        features['Volatility'] = features['Returns'].rolling(window=20).std()
        
        return features
    
    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create all technical indicators and features.
        
        Args:
            df: DataFrame with OHLCV data (must have Open, High, Low, Close, Volume)
            
        Returns:
            DataFrame with all engineered features
        """
        if not all(col in df.columns for col in ['Open', 'High', 'Low', 'Close', 'Volume']):
            raise ValueError("DataFrame must contain Open, High, Low, Close, Volume columns")
        
        features = pd.DataFrame(index=df.index)
        prices = df['Close']
        
        # RSI
        features['RSI'] = self.calculate_rsi(prices)
        
        # MACD
        macd_data = self.calculate_macd(prices)
        features = pd.concat([features, macd_data], axis=1)
        
        # Bollinger Bands
        bb_data = self.calculate_bollinger_bands(prices)
        features = pd.concat([features, bb_data], axis=1)
        
        # Moving Averages
        ma_data = self.calculate_moving_averages(prices)
        features = pd.concat([features, ma_data], axis=1)
        
        # Price features
        price_features = self.calculate_price_features(df)
        features = pd.concat([features, price_features], axis=1)
        
        # Drop rows with NaN values (from indicator calculations)
        features = features.dropna()
        
        return features
    
    def create_sequences(self, features: pd.DataFrame, target: pd.Series, 
                        test_size: float = 0.2) -> Tuple[np.ndarray, np.ndarray, 
                                                        np.ndarray, np.ndarray]:
        """
        Create sequences for LSTM model training.
        
        Args:
            features: DataFrame with engineered features
            target: Series with target values (next day direction: 1 for up, 0 for down)
            test_size: Proportion of data to use for testing (default: 0.2)
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test) arrays
        """
        # Align features and target
        aligned_data = pd.concat([features, target], axis=1).dropna()
        features_aligned = aligned_data[features.columns]
        target_aligned = aligned_data[target.name]
        
        # Convert to numpy arrays
        feature_array = features_aligned.values
        target_array = target_aligned.values
        
        # Create sequences
        X, y = [], []
        for i in range(self.lookback_window, len(feature_array)):
            X.append(feature_array[i - self.lookback_window:i])
            y.append(target_array[i])
        
        X = np.array(X)
        y = np.array(y)
        
        # Train-test split
        split_idx = int(len(X) * (1 - test_size))
        
        X_train = X[:split_idx]
        X_test = X[split_idx:]
        y_train = y[:split_idx]
        y_test = y[split_idx:]
        
        print(f"Created sequences:")
        print(f"  Training samples: {len(X_train)}")
        print(f"  Test samples: {len(X_test)}")
        print(f"  Sequence length: {self.lookback_window}")
        print(f"  Features per timestep: {X_train.shape[2]}")
        
        return X_train, X_test, y_train, y_test
    
    def create_target(self, prices: pd.Series) -> pd.Series:
        """
        Create target variable: next day price direction.
        
        Args:
            prices: Series of closing prices
            
        Returns:
            Series with 1 for price increase, 0 for price decrease
        """
        # Calculate next day return
        next_day_return = prices.shift(-1) / prices - 1
        
        # Create binary target: 1 if price goes up, 0 if price goes down
        target = (next_day_return > 0).astype(int)
        target.name = 'Target'
        
        return target

