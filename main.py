"""
Main Pipeline Script
Orchestrates the entire stock prediction and backtesting pipeline.
"""

import os
import sys
import pandas as pd

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.data_downloader import StockDataDownloader
from src.feature_engineering import FeatureEngineer
from src.model_training import ModelTrainer
from src.strategy import TradingStrategy
from src.backtester import Backtester
from src.visualization import Visualizer


def run_pipeline(ticker: str, period: str = "3y", lookback_window: int = 60,
                epochs: int = 50, batch_size: int = 32, 
                buy_threshold: float = 0.6, sell_threshold: float = 0.4,
                commission: float = 0.001, initial_capital: float = 100000,
                save_model: bool = True, create_plots: bool = True):
    """
    Run the complete stock prediction and backtesting pipeline.
    
    Args:
        ticker: Stock ticker symbol
        period: Data period ('1y', '2y', '3y', etc.)
        lookback_window: LSTM lookback window (default: 60)
        epochs: Number of training epochs (default: 50)
        batch_size: Batch size for training (default: 32)
        buy_threshold: Confidence threshold for BUY signal (default: 0.6)
        sell_threshold: Confidence threshold for SELL signal (default: 0.4)
        commission: Commission rate per trade (default: 0.001)
        initial_capital: Starting capital (default: 100000)
        save_model: Whether to save trained model (default: True)
        create_plots: Whether to create visualization plots (default: True)
    """
    print("=" * 80)
    print(f"STOCK PREDICTION PIPELINE - {ticker}")
    print("=" * 80)
    
    # Create directories
    os.makedirs("data/raw", exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)
    os.makedirs("models/saved_models", exist_ok=True)
    os.makedirs("results/plots", exist_ok=True)
    
    # Step 1: Download Data
    print("\n[STEP 1] Downloading stock data...")
    downloader = StockDataDownloader(ticker, period=period, interval="1d")
    data = downloader.download()
    
    # Save raw data
    data_path = os.path.join("data/raw", f"{ticker}_data.csv")
    downloader.save_to_csv(data_path)
    
    # Step 2: Feature Engineering
    print("\n[STEP 2] Engineering features...")
    engineer = FeatureEngineer(lookback_window=lookback_window)
    features = engineer.create_features(data)
    target = engineer.create_target(data['Close'])
    
    # Save processed features
    features_path = os.path.join("data/processed", f"{ticker}_features.csv")
    features.to_csv(features_path)
    print(f"Features saved to {features_path}")
    
    print(f"Created {len(features.columns)} features")
    print(f"Feature columns: {list(features.columns)}")
    
    # Step 3: Create Sequences
    print("\n[STEP 3] Creating sequences for LSTM...")
    X_train, X_test, y_train, y_test = engineer.create_sequences(features, target, test_size=0.2)
    
    # Step 4: Train Model
    print("\n[STEP 4] Training LSTM model...")
    trainer = ModelTrainer(lookback_window=lookback_window)
    history = trainer.train(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        verbose=1
    )
    
    # Step 5: Evaluate Model
    print("\n[STEP 5] Evaluating model...")
    metrics = trainer.evaluate(X_test, y_test)
    print("\nModel Performance Metrics:")
    print(f"  Accuracy: {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall: {metrics['recall']:.4f}")
    print(f"  F1 Score: {metrics['f1_score']:.4f}")
    
    # Step 6: Generate Predictions
    print("\n[STEP 6] Generating predictions...")
    predictions, confidence = trainer.predict_with_confidence(X_test)
    
    # Step 7: Backtesting
    print("\n[STEP 7] Running backtest...")
    strategy = TradingStrategy(
        buy_threshold=buy_threshold,
        sell_threshold=sell_threshold,
        commission=commission,
        initial_capital=initial_capital
    )
    
    signals = strategy.generate_signals(predictions, confidence)
    
    # Get prices for test period
    test_start_idx = len(data) - len(X_test)
    test_prices = data['Close'].iloc[test_start_idx:]
    test_dates = data.index[test_start_idx:]
    
    backtester = Backtester(strategy)
    results = backtester.backtest(test_prices, signals, dates=test_dates)
    
    # Step 8: Calculate Performance Metrics
    print("\n[STEP 8] Calculating performance metrics...")
    performance_metrics = backtester.calculate_metrics()
    
    print("\n" + "=" * 80)
    print("BACKTEST PERFORMANCE METRICS")
    print("=" * 80)
    for key, value in performance_metrics.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.2f}")
        else:
            print(f"  {key}: {value}")
    
    # Step 9: Save Results
    print("\n[STEP 9] Saving results...")
    
    # Save model
    if save_model:
        model_path = os.path.join("models/saved_models", f"{ticker}_model")
        trainer.save_model(model_path)
    
    # Save backtest results
    results_path = os.path.join("results", f"{ticker}_backtest_results.csv")
    results.to_csv(results_path, index=False)
    print(f"Backtest results saved to {results_path}")
    
    # Save performance metrics
    metrics_df = pd.DataFrame([performance_metrics])
    metrics_path = os.path.join("results", f"{ticker}_performance_metrics.csv")
    metrics_df.to_csv(metrics_path, index=False)
    print(f"Performance metrics saved to {metrics_path}")
    
    # Save trades
    trades = backtester.get_trades()
    if len(trades) > 0:
        trades_path = os.path.join("results", f"{ticker}_trades.csv")
        trades.to_csv(trades_path, index=False)
        print(f"Trade log saved to {trades_path}")
    
    # Step 10: Visualizations
    if create_plots:
        print("\n[STEP 10] Creating visualizations...")
        visualizer = Visualizer(save_dir="results/plots")
        
        # Plot price data
        visualizer.plot_price_data(data, ticker, save=True)
        
        # Plot technical indicators
        visualizer.plot_technical_indicators(data, features, ticker, save=True)
        
        # Plot training history
        visualizer.plot_training_history(history, save=True)
        
        # Plot equity curve
        visualizer.plot_equity_curve(results, ticker, save=True)
        
        # Plot signals
        signal_data = data.iloc[test_start_idx:]
        visualizer.plot_signals(signal_data, signals, ticker, save=True)
        
        # Plot performance metrics
        visualizer.plot_performance_metrics(performance_metrics, ticker, save=True)
        
        # Plot confusion matrix
        visualizer.plot_confusion_matrix(y_test, predictions, save=True)
    
    print("\n" + "=" * 80)
    print("PIPELINE COMPLETE!")
    print("=" * 80)
    
    return {
        'trainer': trainer,
        'backtester': backtester,
        'results': results,
        'metrics': performance_metrics,
        'model_metrics': metrics
    }


if __name__ == "__main__":
    # Example: Run pipeline for AAPL
    tickers = ["AAPL", "MSFT", "NVDA"]  # S&P 500 stocks
    
    print("Stock Price Prediction Pipeline")
    print("=" * 80)
    print("This script will:")
    print("  1. Download 3 years of stock data")
    print("  2. Engineer technical indicators")
    print("  3. Train LSTM model")
    print("  4. Backtest trading strategy")
    print("  5. Generate performance metrics and visualizations")
    print("=" * 80)
    
    # Run for first ticker as example
    ticker = tickers[0]
    
    results = run_pipeline(
        ticker=ticker,
        period="3y",
        lookback_window=60,
        epochs=50,
        batch_size=32,
        buy_threshold=0.50,    # Lowered to 0.5 to generate BUY signals
        sell_threshold=0.50,    # Set to 0.5 to generate SELL signals (balanced)
        commission=0.001,
        initial_capital=100000,
        save_model=True,
        create_plots=True
    )
    
    print(f"\nPipeline completed for {ticker}!")
    print("\nTo run for other tickers, modify the 'tickers' list or call run_pipeline() directly.")

