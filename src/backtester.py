"""
Backtester Module
Simulates trading on historical data.
"""

import pandas as pd
import numpy as np
from typing import Optional
import warnings
warnings.filterwarnings('ignore')

try:
    from .strategy import TradingStrategy
    from .metrics import PerformanceMetrics
except ImportError:
    from strategy import TradingStrategy
    from metrics import PerformanceMetrics


class Backtester:
    """
    Backtests trading strategy on historical data.
    """
    
    def __init__(self, strategy: TradingStrategy):
        """
        Initialize backtester.
        
        Args:
            strategy: TradingStrategy instance
        """
        self.strategy = strategy
        self.results = None
        self.trades = pd.DataFrame()
        
    def backtest(self, prices: pd.Series, signals: np.ndarray,
                dates: Optional[pd.DatetimeIndex] = None) -> pd.DataFrame:
        """
        Run backtest on historical data.
        
        Args:
            prices: Series of closing prices
            signals: Array of trading signals (1=BUY, -1=SELL, 0=HOLD)
            dates: Optional date index for results
            
        Returns:
            DataFrame with backtest results
        """
        if len(prices) != len(signals):
            raise ValueError("Prices and signals must have same length")
        
        # Initialize tracking variables
        position = 0  # 0 = no position, 1 = long position
        capital = self.strategy.initial_capital
        shares = 0
        equity_curve = []
        trades = []
        
        # Align dates
        if dates is None:
            dates = prices.index if isinstance(prices.index, pd.DatetimeIndex) else range(len(prices))
        
        # Run backtest
        for i in range(len(prices)):
            price = prices.iloc[i]
            signal = signals[i]
            date = dates[i] if hasattr(dates, '__getitem__') else dates[i]
            
            # Execute trades based on signals
            if signal == 1 and position == 0:  # BUY signal, no position
                # Buy shares
                shares = capital / price
                commission_cost = capital * self.strategy.commission
                shares = shares * (1 - self.strategy.commission)
                capital = 0
                position = 1
                
                trades.append({
                    'Date': date,
                    'Action': 'BUY',
                    'Price': price,
                    'Shares': shares,
                    'Commission': commission_cost
                })
                
            elif signal == -1 and position == 1:  # SELL signal, have position
                # Sell shares
                capital = shares * price
                commission_cost = capital * self.strategy.commission
                capital = capital * (1 - self.strategy.commission)
                shares = 0
                position = 0
                
                trades.append({
                    'Date': date,
                    'Action': 'SELL',
                    'Price': price,
                    'Shares': shares,
                    'Commission': commission_cost
                })
            
            # Calculate current equity at end of day
            if position == 1:
                current_equity = shares * price
            else:
                current_equity = capital
            
            equity_curve.append(current_equity)
        
        # Close final position if still holding (update last equity value)
        if position == 1:
            final_price = prices.iloc[-1]
            capital = shares * final_price * (1 - self.strategy.commission)
            equity_curve[-1] = capital
        
        # Create results DataFrame
        if isinstance(dates, pd.DatetimeIndex):
            result_dates = dates.tolist()
        else:
            result_dates = list(dates)
        
        results = pd.DataFrame({
            'Date': result_dates,
            'Price': list(prices.values),
            'Signal': list(signals),
            'Equity': equity_curve
        })
        
        results['Returns'] = results['Equity'].pct_change()
        results['Cumulative_Returns'] = (1 + results['Returns']).cumprod() - 1
        
        self.results = results
        self.trades = pd.DataFrame(trades) if trades else pd.DataFrame()
        
        return results
    
    def calculate_metrics(self) -> dict:
        """
        Calculate performance metrics.
        
        Returns:
            Dictionary with performance metrics
        """
        if self.results is None:
            raise ValueError("No backtest results. Run backtest() first.")
        
        return PerformanceMetrics.calculate_all_metrics(
            self.results, 
            self.trades, 
            self.strategy.initial_capital
        )
    
    def get_results(self) -> pd.DataFrame:
        """
        Get backtest results.
        
        Returns:
            DataFrame with backtest results
        """
        if self.results is None:
            raise ValueError("No backtest results. Run backtest() first.")
        return self.results.copy()
    
    def get_trades(self) -> pd.DataFrame:
        """
        Get trade log.
        
        Returns:
            DataFrame with trade details
        """
        return self.trades.copy()

