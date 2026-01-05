"""
Data Downloader Module
Downloads historical stock data using yfinance library.
"""

import yfinance as yf
import pandas as pd
from typing import List, Optional
import os


class StockDataDownloader:
    """
    Downloads and manages stock market data from Yahoo Finance.
    
    Attributes:
        ticker (str): Stock ticker symbol
        period (str): Data period ('1y', '2y', '3y', etc.)
        interval (str): Data interval ('1d', '1h', etc.)
    """
    
    def __init__(self, ticker: str, period: str = "3y", interval: str = "1d"):
        """
        Initialize the data downloader.
        
        Args:
            ticker: Stock ticker symbol (e.g., 'AAPL')
            period: Time period for data ('1y', '2y', '3y', 'max')
            interval: Data interval ('1d' for daily, '1h' for hourly)
        """
        self.ticker = ticker.upper()
        self.period = period
        self.interval = interval
        self.data = None
        
    def download(self) -> pd.DataFrame:
        """
        Download historical stock data.
        
        Returns:
            DataFrame with OHLCV data
            
        Raises:
            Exception: If download fails or ticker is invalid
        """
        try:
            print(f"Downloading {self.period} of {self.interval} data for {self.ticker}...")
            stock = yf.Ticker(self.ticker)
            self.data = stock.history(period=self.period, interval=self.interval)
            
            if self.data.empty:
                raise ValueError(f"No data retrieved for ticker {self.ticker}")
            
            # Clean column names (remove spaces)
            self.data.columns = [col.replace(' ', '') for col in self.data.columns]
            
            # Ensure we have required columns
            required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
            missing_cols = [col for col in required_cols if col not in self.data.columns]
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")
            
            # Remove any rows with NaN values
            initial_rows = len(self.data)
            self.data = self.data.dropna()
            removed_rows = initial_rows - len(self.data)
            
            if removed_rows > 0:
                print(f"Warning: Removed {removed_rows} rows with missing data")
            
            print(f"Successfully downloaded {len(self.data)} rows of data")
            print(f"Date range: {self.data.index[0]} to {self.data.index[-1]}")
            
            return self.data
            
        except Exception as e:
            raise Exception(f"Error downloading data for {self.ticker}: {str(e)}")
    
    def save_to_csv(self, filepath: str) -> None:
        """
        Save downloaded data to CSV file.
        
        Args:
            filepath: Path to save the CSV file
        """
        if self.data is None:
            raise ValueError("No data to save. Please download data first.")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
        
        self.data.to_csv(filepath)
        print(f"Data saved to {filepath}")
    
    def load_from_csv(self, filepath: str) -> pd.DataFrame:
        """
        Load data from CSV file.
        
        Args:
            filepath: Path to the CSV file
            
        Returns:
            DataFrame with stock data
        """
        try:
            self.data = pd.read_csv(filepath, index_col=0, parse_dates=True)
            print(f"Data loaded from {filepath}")
            return self.data
        except Exception as e:
            raise Exception(f"Error loading data from {filepath}: {str(e)}")
    
    def get_data(self) -> pd.DataFrame:
        """
        Get the downloaded data.
        
        Returns:
            DataFrame with stock data
        """
        if self.data is None:
            raise ValueError("No data available. Please download data first.")
        return self.data.copy()


def download_multiple_stocks(tickers: List[str], period: str = "3y", 
                            interval: str = "1d", save_dir: str = "data/raw") -> dict:
    """
    Download data for multiple stocks.
    
    Args:
        tickers: List of ticker symbols
        period: Time period for data
        interval: Data interval
        save_dir: Directory to save CSV files
        
    Returns:
        Dictionary mapping ticker symbols to DataFrames
    """
    os.makedirs(save_dir, exist_ok=True)
    data_dict = {}
    
    for ticker in tickers:
        try:
            downloader = StockDataDownloader(ticker, period, interval)
            data = downloader.download()
            data_dict[ticker] = data
            
            # Save to CSV
            filepath = os.path.join(save_dir, f"{ticker}_data.csv")
            downloader.save_to_csv(filepath)
            
        except Exception as e:
            print(f"Failed to download {ticker}: {str(e)}")
            continue
    
    return data_dict

