"""
Metrics Module
Calculates performance metrics (Sharpe ratio, drawdown, etc.).
"""

import pandas as pd
import numpy as np
from typing import Dict


class PerformanceMetrics:
    """
    Calculates various performance metrics for backtest results.
    """
    
    @staticmethod
    def calculate_sharpe_ratio(returns: np.ndarray, risk_free_rate: float = 0.0) -> float:
        """
        Calculate Sharpe ratio.
        
        Args:
            returns: Array of returns
            risk_free_rate: Risk-free rate (default: 0.0)
            
        Returns:
            Sharpe ratio
        """
        if len(returns) == 0 or returns.std() == 0:
            return 0.0
        
        excess_returns = returns - risk_free_rate / 252  # Daily risk-free rate
        return (excess_returns.mean() / returns.std()) * np.sqrt(252)
    
    @staticmethod
    def calculate_max_drawdown(equity: np.ndarray) -> float:
        """
        Calculate maximum drawdown.
        
        Args:
            equity: Array of equity values
            
        Returns:
            Maximum drawdown as percentage
        """
        if len(equity) == 0:
            return 0.0
        
        cumulative = equity / equity[0]
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        
        return drawdown.min() * 100
    
    @staticmethod
    def calculate_returns(equity: np.ndarray, days: int = None) -> Dict[str, float]:
        """
        Calculate total and annualized returns.
        
        Args:
            equity: Array of equity values
            days: Number of trading days (default: len(equity))
            
        Returns:
            Dictionary with return metrics
        """
        if len(equity) == 0:
            return {'total_return': 0.0, 'annualized_return': 0.0}
        
        total_return = (equity[-1] / equity[0] - 1) * 100
        
        if days is None:
            days = len(equity)
        
        years = days / 252  # Trading days per year
        annualized_return = ((equity[-1] / equity[0]) ** (1 / years) - 1) * 100 if years > 0 else 0
        
        return {
            'total_return': total_return,
            'annualized_return': annualized_return
        }
    
    @staticmethod
    def calculate_win_rate(trades: pd.DataFrame) -> float:
        """
        Calculate win rate from trades.
        
        Args:
            trades: DataFrame with trade details
            
        Returns:
            Win rate as percentage
        """
        if len(trades) == 0:
            return 0.0
        
        buy_trades = trades[trades['Action'] == 'BUY']
        sell_trades = trades[trades['Action'] == 'SELL']
        
        if len(buy_trades) == 0 or len(sell_trades) == 0:
            return 0.0
        
        # Match buy/sell pairs
        profitable_trades = 0
        total_trades = min(len(buy_trades), len(sell_trades))
        
        for i in range(total_trades):
            if i < len(sell_trades):
                buy_price = buy_trades.iloc[i]['Price']
                sell_price = sell_trades.iloc[i]['Price']
                if sell_price > buy_price:
                    profitable_trades += 1
        
        return (profitable_trades / total_trades * 100) if total_trades > 0 else 0.0
    
    @staticmethod
    def calculate_all_metrics(results: pd.DataFrame, trades: pd.DataFrame,
                             initial_capital: float) -> Dict[str, float]:
        """
        Calculate all performance metrics.
        
        Args:
            results: DataFrame with backtest results
            trades: DataFrame with trade details
            initial_capital: Starting capital
            
        Returns:
            Dictionary with all performance metrics
        """
        equity = results['Equity'].values
        returns = results['Returns'].dropna().values
        
        # Returns
        return_metrics = PerformanceMetrics.calculate_returns(equity, days=len(results))
        
        # Sharpe Ratio
        sharpe_ratio = PerformanceMetrics.calculate_sharpe_ratio(returns)
        
        # Maximum Drawdown
        max_drawdown = PerformanceMetrics.calculate_max_drawdown(equity)
        
        # Win Rate
        win_rate = PerformanceMetrics.calculate_win_rate(trades)
        
        # Number of trades
        num_trades = len(trades)
        
        metrics = {
            'Total_Return_%': return_metrics['total_return'],
            'Annualized_Return_%': return_metrics['annualized_return'],
            'Sharpe_Ratio': sharpe_ratio,
            'Max_Drawdown_%': max_drawdown,
            'Win_Rate_%': win_rate,
            'Number_of_Trades': num_trades,
            'ROI_%': return_metrics['total_return'],
            'Initial_Capital': initial_capital,
            'Final_Capital': equity[-1]
        }
        
        return metrics

