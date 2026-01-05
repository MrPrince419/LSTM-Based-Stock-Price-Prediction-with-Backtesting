"""
Visualization Module
Creates charts and plots for model performance and backtest results.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Optional, Dict
import os

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)


class Visualizer:
    """
    Creates visualizations for stock prediction and backtesting results.
    """
    
    def __init__(self, save_dir: str = "results/plots"):
        """
        Initialize visualizer.
        
        Args:
            save_dir: Directory to save plots
        """
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
    
    def plot_price_data(self, data: pd.DataFrame, ticker: str, 
                       save: bool = True) -> None:
        """Plot historical price data with volume."""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), 
                                       gridspec_kw={'height_ratios': [3, 1]})
        
        ax1.plot(data.index, data['Close'], label='Close Price', linewidth=2)
        ax1.fill_between(data.index, data['Low'], data['High'], 
                         alpha=0.3, label='High-Low Range')
        ax1.set_title(f'{ticker} Stock Price History', fontsize=16, fontweight='bold')
        ax1.set_ylabel('Price ($)', fontsize=12)
        ax1.legend(loc='upper left')
        ax1.grid(True, alpha=0.3)
        
        ax2.bar(data.index, data['Volume'], alpha=0.6, color='steelblue')
        ax2.set_title('Trading Volume', fontsize=14)
        ax2.set_xlabel('Date', fontsize=12)
        ax2.set_ylabel('Volume', fontsize=12)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save:
            filepath = os.path.join(self.save_dir, f'{ticker}_price_data.png')
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {filepath}")
        
        plt.show()
    
    def plot_technical_indicators(self, data: pd.DataFrame, features: pd.DataFrame,
                                 ticker: str, save: bool = True) -> None:
        """Plot technical indicators."""
        fig, axes = plt.subplots(4, 1, figsize=(14, 12))
        
        aligned_data = pd.concat([data['Close'], features], axis=1).dropna()
        
        axes[0].plot(aligned_data.index, aligned_data['Close'], 
                    label='Close Price', linewidth=2)
        if 'SMA_20' in aligned_data.columns:
            axes[0].plot(aligned_data.index, aligned_data['SMA_20'], 
                        label='SMA 20', alpha=0.7)
        if 'SMA_50' in aligned_data.columns:
            axes[0].plot(aligned_data.index, aligned_data['SMA_50'], 
                        label='SMA 50', alpha=0.7)
        axes[0].set_title('Price with Moving Averages', fontsize=12, fontweight='bold')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        if 'RSI' in aligned_data.columns:
            axes[1].plot(aligned_data.index, aligned_data['RSI'], 
                       label='RSI', color='purple', linewidth=2)
            axes[1].axhline(y=70, color='r', linestyle='--', alpha=0.5, label='Overbought')
            axes[1].axhline(y=30, color='g', linestyle='--', alpha=0.5, label='Oversold')
            axes[1].set_title('Relative Strength Index (RSI)', fontsize=12, fontweight='bold')
            axes[1].set_ylabel('RSI', fontsize=10)
            axes[1].set_ylim(0, 100)
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)
        
        if 'MACD' in aligned_data.columns:
            axes[2].plot(aligned_data.index, aligned_data['MACD'], 
                        label='MACD', linewidth=2)
            if 'MACD_Signal' in aligned_data.columns:
                axes[2].plot(aligned_data.index, aligned_data['MACD_Signal'], 
                           label='Signal', linewidth=1.5)
            axes[2].axhline(y=0, color='black', linestyle='-', alpha=0.3)
            axes[2].set_title('MACD', fontsize=12, fontweight='bold')
            axes[2].set_ylabel('MACD', fontsize=10)
            axes[2].legend()
            axes[2].grid(True, alpha=0.3)
        
        if 'BB_Upper' in aligned_data.columns:
            axes[3].plot(aligned_data.index, aligned_data['Close'], 
                        label='Close Price', linewidth=2)
            axes[3].plot(aligned_data.index, aligned_data['BB_Upper'], 
                        label='Upper Band', alpha=0.5, linestyle='--')
            axes[3].plot(aligned_data.index, aligned_data['BB_Lower'], 
                        label='Lower Band', alpha=0.5, linestyle='--')
            axes[3].fill_between(aligned_data.index, aligned_data['BB_Upper'], 
                                aligned_data['BB_Lower'], alpha=0.1)
            axes[3].set_title('Bollinger Bands', fontsize=12, fontweight='bold')
            axes[3].set_xlabel('Date', fontsize=10)
            axes[3].set_ylabel('Price', fontsize=10)
            axes[3].legend()
            axes[3].grid(True, alpha=0.3)
        
        plt.suptitle(f'{ticker} Technical Indicators', fontsize=16, fontweight='bold', y=0.995)
        plt.tight_layout()
        
        if save:
            filepath = os.path.join(self.save_dir, f'{ticker}_technical_indicators.png')
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {filepath}")
        
        plt.show()
    
    def plot_training_history(self, history: Dict, save: bool = True) -> None:
        """Plot model training history."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        epochs = range(1, len(history['loss']) + 1)
        
        ax1.plot(epochs, history['loss'], 'b-', label='Training Loss', linewidth=2)
        if 'val_loss' in history:
            ax1.plot(epochs, history['val_loss'], 'r-', label='Validation Loss', linewidth=2)
        ax1.set_title('Model Loss', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Epoch', fontsize=12)
        ax1.set_ylabel('Loss', fontsize=12)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        ax2.plot(epochs, history['accuracy'], 'b-', label='Training Accuracy', linewidth=2)
        if 'val_accuracy' in history:
            ax2.plot(epochs, history['val_accuracy'], 'r-', label='Validation Accuracy', linewidth=2)
        ax2.set_title('Model Accuracy', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Epoch', fontsize=12)
        ax2.set_ylabel('Accuracy', fontsize=12)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save:
            filepath = os.path.join(self.save_dir, 'training_history.png')
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {filepath}")
        
        plt.show()
    
    def plot_equity_curve(self, results: pd.DataFrame, ticker: str,
                         save: bool = True) -> None:
        """Plot equity curve from backtest results."""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), 
                                       gridspec_kw={'height_ratios': [2, 1]})
        
        ax1.plot(results['Date'], results['Equity'], linewidth=2, 
                label='Portfolio Equity', color='steelblue')
        ax1.axhline(y=results['Equity'].iloc[0], color='r', 
                   linestyle='--', alpha=0.5, label='Initial Capital')
        ax1.set_title(f'{ticker} Backtest Equity Curve', fontsize=16, fontweight='bold')
        ax1.set_ylabel('Equity ($)', fontsize=12)
        ax1.legend(loc='upper left')
        ax1.grid(True, alpha=0.3)
        
        ax2.plot(results['Date'], results['Cumulative_Returns'] * 100, 
                linewidth=2, label='Cumulative Returns', color='green')
        ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax2.set_title('Cumulative Returns', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Date', fontsize=12)
        ax2.set_ylabel('Returns (%)', fontsize=12)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save:
            filepath = os.path.join(self.save_dir, f'{ticker}_equity_curve.png')
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {filepath}")
        
        plt.show()
    
    def plot_signals(self, data: pd.DataFrame, signals: np.ndarray,
                    ticker: str, save: bool = True) -> None:
        """Plot trading signals on price chart."""
        fig, ax = plt.subplots(figsize=(14, 7))
        
        ax.plot(data.index, data['Close'], label='Close Price', 
               linewidth=2, color='black', alpha=0.7)
        
        buy_indices = np.where(signals == 1)[0]
        if len(buy_indices) > 0:
            buy_dates = data.index[buy_indices]
            buy_prices = data['Close'].iloc[buy_indices]
            ax.scatter(buy_dates, buy_prices, color='green', marker='^', 
                      s=100, label='BUY Signal', zorder=5)
        
        sell_indices = np.where(signals == -1)[0]
        if len(sell_indices) > 0:
            sell_dates = data.index[sell_indices]
            sell_prices = data['Close'].iloc[sell_indices]
            ax.scatter(sell_dates, sell_prices, color='red', marker='v', 
                      s=100, label='SELL Signal', zorder=5)
        
        ax.set_title(f'{ticker} Trading Signals', fontsize=16, fontweight='bold')
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Price ($)', fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save:
            filepath = os.path.join(self.save_dir, f'{ticker}_signals.png')
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {filepath}")
        
        plt.show()
    
    def plot_performance_metrics(self, metrics: Dict, ticker: str,
                                save: bool = True) -> None:
        """Plot performance metrics as bar chart."""
        key_metrics = {
            'Total Return (%)': metrics.get('Total_Return_%', 0),
            'Sharpe Ratio': metrics.get('Sharpe_Ratio', 0),
            'Max Drawdown (%)': metrics.get('Max_Drawdown_%', 0),
            'Win Rate (%)': metrics.get('Win_Rate_%', 0)
        }
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        colors = ['green' if v > 0 else 'red' for v in key_metrics.values()]
        bars = ax.bar(key_metrics.keys(), key_metrics.values(), color=colors, alpha=0.7)
        
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.2f}',
                   ha='center', va='bottom' if height > 0 else 'top', fontsize=11)
        
        ax.set_title(f'{ticker} Backtest Performance Metrics', 
                    fontsize=16, fontweight='bold')
        ax.set_ylabel('Value', fontsize=12)
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        if save:
            filepath = os.path.join(self.save_dir, f'{ticker}_performance_metrics.png')
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {filepath}")
        
        plt.show()
    
    def plot_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray,
                             save: bool = True) -> None:
        """Plot confusion matrix."""
        from sklearn.metrics import confusion_matrix
        
        cm = confusion_matrix(y_true, y_pred)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                   xticklabels=['Down', 'Up'], yticklabels=['Down', 'Up'])
        ax.set_title('Confusion Matrix', fontsize=16, fontweight='bold')
        ax.set_ylabel('True Label', fontsize=12)
        ax.set_xlabel('Predicted Label', fontsize=12)
        
        plt.tight_layout()
        
        if save:
            filepath = os.path.join(self.save_dir, 'confusion_matrix.png')
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {filepath}")
        
        plt.show()

