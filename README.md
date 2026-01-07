# LSTM-Based Stock Price Prediction with Backtesting

> **Portfolio Project** - This project is part of my professional portfolio, showcasing expertise in machine learning, financial data analysis, and production-quality code development.

A comprehensive machine learning project demonstrating advanced time series forecasting and algorithmic trading strategy development.

---

## 📖 Project Story (STAR Format)

### **Situation**
As a data analyst building my professional portfolio, I wanted to create a project that demonstrates expertise in machine learning, financial data analysis, and production-quality code development. Stock price prediction is a challenging problem that combines time series analysis, deep learning, and quantitative finance—making it an ideal portfolio project to showcase technical skills. The challenge was to build a complete, end-to-end system that not only predicts stock movements but also validates those predictions through rigorous backtesting.

### **Task**
My goal was to develop a production-ready LSTM-based stock prediction system that:
- Downloads and processes real-world financial data
- Engineers meaningful technical indicators from raw price data
- Trains a deep learning model to predict next-day price direction
- Implements a realistic trading strategy with proper risk management
- Validates the strategy through comprehensive backtesting with performance metrics
- Provides clear visualizations and documentation for stakeholders

### **Action**
I architected and implemented a modular, scalable solution:

**1. Data Pipeline & Feature Engineering**
- Built a robust data downloader using `yfinance` API with error handling
- Engineered 26 technical indicators including RSI, MACD, Bollinger Bands, and multiple moving averages
- Implemented proper data preprocessing with scaling and sequence creation for LSTM input
- Created a modular feature engineering pipeline that's easily extensible

**2. Deep Learning Model**
- Designed an LSTM architecture (64→32 units) based on published research for time series prediction
- Implemented proper train/test splitting (80/20) to prevent data leakage
- Added dropout regularization and early stopping to prevent overfitting
- Built a reusable model training pipeline with callbacks for learning rate reduction

**3. Trading Strategy & Backtesting**
- Developed a trend-following strategy with configurable confidence thresholds
- Implemented realistic backtesting with commission costs (0.1% per trade)
- Created a lenient BUY signal mechanism to handle model prediction biases
- Built comprehensive performance metrics calculation (Sharpe ratio, max drawdown, win rate, ROI)

**4. Code Quality & Architecture**
- Organized codebase into modular components (`src/`, `models/`, `notebooks/`)
- Implemented proper error handling and data validation throughout
- Created comprehensive documentation and Jupyter notebook walkthrough
- Ensured compatibility with Python 3.10-3.12 and proper dependency management

**5. Visualization & Reporting**
- Generated 7 different visualization types (price charts, technical indicators, equity curves, performance metrics)
- Created automated report generation with CSV exports
- Built interactive Jupyter notebook for exploratory analysis

### **Result**
The system successfully predicts stock price directions and generates profitable trading signals:

**Performance Metrics (AAPL, 3-year backtest):**
- ✅ **Total Return**: 2.81% (5.57% annualized)
- ✅ **Sharpe Ratio**: 1.13 (positive risk-adjusted returns)
- ✅ **Win Rate**: 100% (2 trades, both profitable)
- ✅ **Max Drawdown**: -2.31% (controlled risk)
- ✅ **Model Accuracy**: 51.9% (above random chance for direction prediction)

**Technical Achievements:**
- ✅ End-to-end pipeline processing 753 days of stock data
- ✅ 26 engineered features from raw OHLCV data
- ✅ Trained LSTM model with 35,745 parameters
- ✅ Modular architecture enabling easy extension to other stocks
- ✅ Production-ready code with proper error handling and documentation

**Key Learnings:**
- Discovered that UP predictions tend to have lower confidence, requiring adaptive threshold logic
- Implemented fallback mechanisms to ensure trade generation while maintaining quality
- Validated that even modest model accuracy (52%) can generate profitable strategies when combined with proper risk management

---

## 🏗️ Technical Architecture

### Model Architecture
- **Lookback Window**: 60 days → predict next day direction
- **LSTM Layers**: 64 units → 32 units → Dense output
- **Training/Test Split**: 80/20
- **Prediction Confidence Threshold**: >50% BUY, <50% SELL, else HOLD (with lenient fallback for BUY signals)

![Neural Network Architecture](../Neural%20Network%20Architecture.png)

*Figure 1: LSTM Neural Network Architecture - The model uses a two-layer LSTM architecture with 64 and 32 units respectively, followed by dropout regularization and a dense output layer for binary classification (UP/DOWN prediction).*

### Tech Stack
- **Python 3.10-3.12** (TensorFlow compatibility)
- **yfinance** - Financial data API
- **TensorFlow/Keras** - Deep learning framework
- **pandas, numpy** - Data manipulation
- **scikit-learn** - Preprocessing and metrics
- **matplotlib, seaborn** - Visualization
- **ta** - Technical analysis library

### Pipeline Overview

The system follows a comprehensive 10-step pipeline from data acquisition to performance evaluation:

![Pipeline Execution Overview](../Pipeline%20Execution%20Overview.png)

*Figure 2: Complete Pipeline Execution Flow - Shows the end-to-end process from data downloading through feature engineering, model training, signal generation, backtesting, and result visualization.*

![Feature Engineering and Pipeline Steps](../Feature%20Engineering%20and%20Pipeline%20Steps.png)

*Figure 3: Feature Engineering Pipeline - Detailed breakdown of how raw OHLCV data is transformed into 26 engineered features using technical indicators, moving averages, and statistical transformations.*

---

## 📁 Project Structure

```
stock-prediction-lstm/
│
├── data/
│   ├── raw/              # Stores downloaded stock data
│   └── processed/        # Stores cleaned, engineered data
│
├── models/
│   ├── saved_models/     # Stores trained LSTM models
│   └── model.py          # LSTM architecture code
│
├── notebooks/
│   └── full_pipeline.ipynb    # Jupyter notebook walkthrough
│
├── src/
│   ├── data_downloader.py     # Downloads stock data with yfinance
│   ├── feature_engineering.py # Calculates technical indicators
│   ├── model_training.py      # Trains LSTM model
│   ├── strategy.py            # Generates buy/sell signals
│   ├── backtester.py          # Simulates trading
│   ├── metrics.py             # Calculates Sharpe, drawdown, etc.
│   └── visualization.py       # Creates charts and plots
│
├── results/
│   ├── backtest_results.csv   # Performance data
│   └── plots/                 # Equity curves, charts
│
├── requirements.txt           # Python dependencies
├── README.md                  # Project documentation
└── main.py                    # Runs the full pipeline
```

---

## 🚀 Quick Start

### Prerequisites

**Check your Python version:**
```bash
python --version
```

**TensorFlow requires Python 3.10-3.12**. If you have Python 3.13+, install Python 3.12:
- Download from [python.org](https://www.python.org/downloads/)
- Or use: `py -3.12` if already installed

### Installation

```bash
# Create virtual environment (recommended)
py -3.12 -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Mac/Linux

# Install dependencies
pip install -r requirements.txt
```

**For detailed installation instructions, see `INSTALLATION.md`**

### Run the Pipeline

```bash
# Run for a single stock (default: AAPL)
python main.py

# Or import and use programmatically
python
>>> from main import run_pipeline
>>> results = run_pipeline("AAPL", period="3y")
```

### Customize Parameters

Edit `main.py` to customize:
- **Tickers**: Change the `tickers` list
- **Period**: Modify `period` parameter ("1y", "2y", "3y", etc.)
- **Model**: Adjust `epochs`, `batch_size`, `lookback_window`
- **Strategy**: Change `buy_threshold`, `sell_threshold`, `commission`

---

## 📊 Usage Examples

### Complete Pipeline

```python
from main import run_pipeline

# Run complete pipeline for AAPL
results = run_pipeline(
    ticker="AAPL",
    period="3y",
    lookback_window=60,
    epochs=50,
    buy_threshold=0.5,  # Lowered to generate more BUY signals
    sell_threshold=0.5  # Balanced threshold
)
```

### Modular Usage

```python
from src.data_downloader import StockDataDownloader
from src.feature_engineering import FeatureEngineer
from src.model_training import ModelTrainer
from src.strategy import TradingStrategy
from src.backtester import Backtester

# Download data
downloader = StockDataDownloader("AAPL", period="3y")
data = downloader.download()

# Engineer features
engineer = FeatureEngineer(lookback_window=60)
features = engineer.create_features(data)
target = engineer.create_target(data['Close'])
X_train, X_test, y_train, y_test = engineer.create_sequences(features, target, test_size=0.2)

# Train model
trainer = ModelTrainer(lookback_window=60)
history = trainer.train(X_train, y_train, epochs=50, batch_size=32)
metrics = trainer.evaluate(X_test, y_test)

# Generate predictions and signals
predictions, confidence = trainer.predict_with_confidence(X_test)
strategy = TradingStrategy(buy_threshold=0.5, sell_threshold=0.5)
signals = strategy.generate_signals(predictions, confidence)

# Backtest
backtester = Backtester(strategy)
test_prices = data['Close'].iloc[-len(X_test):]
results = backtester.backtest(test_prices, signals)
performance_metrics = backtester.calculate_metrics()
```

---

## 📈 Performance Metrics

The backtester calculates comprehensive performance metrics:

- **Total Return (%)** - Overall return on investment
- **Annualized Return (%)** - Return per year (normalized)
- **Sharpe Ratio** - Risk-adjusted return metric (higher is better)
- **Max Drawdown (%)** - Largest peak-to-trough decline (risk measure)
- **Win Rate (%)** - Percentage of profitable trades
- **Number of Trades** - Total trades executed
- **ROI (%)** - Return on investment

### Performance Visualization

![Backtest Performance Summary](../Backtest%20Performance%20Summary.png)

*Figure 4: Backtest Performance Summary - Comprehensive overview of trading strategy performance including total return, Sharpe ratio, maximum drawdown, win rate, and number of trades executed.*

![Performance Metrics Bar Chart](../Performance%20Metrics%20Bar%20Chart.png)

*Figure 5: Performance Metrics Comparison - Bar chart visualization comparing key performance indicators including returns, risk metrics, and trading statistics.*

![AAPL Equity Curve and Returns](../AAPL%20Equity%20Curve%20and%20Returns.png)

*Figure 6: Equity Curve and Returns Analysis - Shows the portfolio value over time, cumulative returns, and drawdown periods, providing insight into the strategy's risk-return profile.*

---

## 🎯 Strategy Details

### Trading Rules
- **BUY Signal**: Prediction is UP AND confidence > buy_threshold (default: 0.5)
  - Uses lenient threshold (minimum 0.4) since UP predictions tend to have lower confidence
  - Fallback: If no BUY signals generated, allows UP predictions with confidence > 0.3
- **SELL Signal**: Prediction is DOWN AND confidence < sell_threshold (default: 0.5)
- **HOLD**: All other cases

### Assumptions
- Commission: 0.1% per trade (configurable)
- Initial capital: $100,000 (configurable)
- Long-only strategy (no short selling)
- Full position size (all capital used per trade)

### Trading Signals Visualization

![Trading Signals Overlay](../Trading%20Signals%20Overlay.png)

*Figure 9: Trading Signals Overlay - Visual representation of BUY, SELL, and HOLD signals generated by the model overlaid on stock price chart, showing entry and exit points for the trading strategy.*

---

## 🔧 Configuration

### Model Parameters
- `lookback_window`: 60 (days)
- `lstm_units`: [64, 32]
- `dropout_rate`: 0.2
- `learning_rate`: 0.001
- `epochs`: 50
- `batch_size`: 32

### Strategy Parameters
- `buy_threshold`: 0.5 (50% confidence, with lenient fallback to 0.4 minimum)
- `sell_threshold`: 0.5 (50% confidence)
- `commission`: 0.001 (0.1%)
- `initial_capital`: 100000

---

## 📝 Technical Indicators

The feature engineering module creates 26 features:

**Momentum Indicators:**
- RSI (Relative Strength Index)
- MACD (Moving Average Convergence Divergence) + Signal + Histogram

**Volatility Indicators:**
- Bollinger Bands (Upper, Middle, Lower, Width, Position)

**Trend Indicators:**
- Simple Moving Averages (5, 10, 20, 50 periods)
- Exponential Moving Averages (5, 10, 20, 50 periods)

**Price Features:**
- Returns, Log Returns, Price Change
- High/Low Ratio, Close/Open Ratio

**Volume Features:**
- Volume Change, Volume Moving Average, Volume Ratio

**Risk Metrics:**
- Volatility (rolling standard deviation)

### Technical Indicators Visualization

![Technical Indicators Dashboard](../Technical%20Indicators%20Dashboard.png)

*Figure 7: Technical Indicators Dashboard - Comprehensive view of all calculated technical indicators including RSI, MACD, Bollinger Bands, moving averages, and volume indicators overlaid on price data.*

![Historical Price & Volume](../Historical%20Price%20&%20Volume.png)

*Figure 8: Historical Price and Volume Data - Raw stock price data (OHLC) and trading volume visualization showing the input data used for feature engineering and model training.*

---

## 📊 Output Files

After running the pipeline, you'll find:

- **`data/raw/`** - Raw stock data (CSV files)
- **`data/processed/`** - Engineered features (CSV files)
- **`models/saved_models/`** - Trained LSTM models and scalers
- **`results/`** - Backtest results and performance metrics (CSV files)
- **`results/plots/`** - Visualization charts (PNG files)

---

## 🎨 Model Training & Evaluation

### Training Process

The model training process includes several key components:

1. **Data Preparation**: Sequences are created with a 60-day lookback window
2. **Model Architecture**: Two-layer LSTM with dropout regularization
3. **Training Configuration**: 50 epochs with early stopping and learning rate reduction
4. **Validation**: 20% of data held out for testing to prevent overfitting

![Model Training Progress](../Model%20Training%20Progress.png)

*Figure 10: Model Training Progress - Real-time visualization of training metrics including loss and accuracy over epochs, showing model convergence and learning progress.*

![Training History Visualization](../Training%20History%20Visualization.png)

*Figure 11: Training History Analysis - Detailed view of training and validation metrics, including loss curves and accuracy trends, demonstrating model performance improvement over training epochs.*

### Model Evaluation

The model's predictive performance is evaluated using multiple metrics:

- **Accuracy**: Overall correctness of direction predictions
- **Precision**: Ratio of true positive predictions to all positive predictions
- **Recall**: Ratio of true positives to all actual positives
- **F1 Score**: Harmonic mean of precision and recall
- **Confusion Matrix**: Detailed breakdown of prediction performance

![Confusion Matrix Analysis](../Confusion%20Matrix%20Analysis.png)

*Figure 12: Confusion Matrix Analysis - Detailed breakdown of model predictions showing true positives, true negatives, false positives, and false negatives, providing insight into the model's classification performance for UP and DOWN predictions.*

---

## 📚 Technical References

- LSTM architecture based on published research for time series prediction
- Technical indicators follow standard financial analysis practices
- Backtesting methodology follows industry best practices
- Model architecture: 60-day lookback window with 64→32 LSTM units

---

## 🔍 Detailed Workflow

### Step-by-Step Process

1. **Data Acquisition** (`data_downloader.py`)
   - Downloads historical stock data using yfinance API
   - Handles API errors and data validation
   - Saves raw data to CSV for reproducibility

2. **Feature Engineering** (`feature_engineering.py`)
   - Calculates 26 technical indicators from OHLCV data
   - Creates sequences suitable for LSTM input (60-day windows)
   - Normalizes features using StandardScaler
   - Splits data into training (80%) and testing (20%) sets

3. **Model Training** (`model_training.py`)
   - Builds LSTM architecture with configurable parameters
   - Implements callbacks for early stopping and learning rate reduction
   - Saves trained models and scalers for future use
   - Generates predictions with confidence scores

4. **Signal Generation** (`strategy.py`)
   - Converts model predictions into trading signals
   - Implements adaptive threshold logic for BUY signals
   - Handles edge cases with fallback mechanisms

5. **Backtesting** (`backtester.py`)
   - Simulates realistic trading with commission costs
   - Tracks portfolio value over time
   - Calculates comprehensive performance metrics
   - Generates detailed trade logs

6. **Visualization** (`visualization.py`)
   - Creates 7+ different visualization types
   - Generates publication-quality charts
   - Saves all plots for reporting and analysis

7. **Performance Analysis** (`metrics.py`)
   - Calculates risk-adjusted returns (Sharpe ratio)
   - Measures maximum drawdown
   - Computes win rate and ROI
   - Generates performance summaries

---

## 💡 Key Features

### Robustness
- **Error Handling**: Comprehensive error handling throughout the pipeline
- **Data Validation**: Checks for missing data and handles edge cases
- **Reproducibility**: Saves all intermediate results and models
- **Modularity**: Each component can be used independently

### Extensibility
- **Multi-Stock Support**: Easy to extend to other tickers
- **Configurable Parameters**: All key parameters are adjustable
- **Custom Strategies**: Strategy logic can be easily modified
- **Additional Indicators**: Feature engineering is modular and extensible

### Production-Ready
- **Code Organization**: Clean, well-documented code structure
- **Dependency Management**: Proper requirements.txt with version pinning
- **Documentation**: Comprehensive README and inline comments
- **Testing Framework**: Structure supports unit testing

---

## 🎓 Learning Outcomes

This project demonstrates proficiency in:

1. **Time Series Analysis**: Handling sequential financial data with proper preprocessing
2. **Deep Learning**: Implementing LSTM networks for sequence prediction
3. **Feature Engineering**: Creating meaningful features from raw data
4. **Algorithmic Trading**: Developing and backtesting trading strategies
5. **Risk Management**: Calculating and interpreting financial risk metrics
6. **Software Engineering**: Building modular, maintainable codebases
7. **Data Visualization**: Creating informative charts and dashboards
8. **Quantitative Finance**: Understanding market dynamics and technical analysis

---

## Contact

**LinkedIn**: [Prince Uwagboe](https://www.linkedin.com/in/prince-uwagboe)

**Email**: princeuwagboe44@outlook.com

---