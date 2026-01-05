# Installation Guide

## Python Version Requirements

**IMPORTANT**: This project requires **Python 3.10, 3.11, or 3.12**.

TensorFlow (the deep learning framework used) does not support Python 3.13 or 3.14.

### Check Your Python Version

```bash
python --version
```

If you have Python 3.13 or 3.14, you have two options:

### Option 1: Install Python 3.12 (Recommended)

1. Download Python 3.12 from [python.org](https://www.python.org/downloads/)
2. Install it (you can have multiple Python versions installed)
3. Create a virtual environment with Python 3.12:
   ```bash
   py -3.12 -m venv venv
   venv\Scripts\activate
   ```

### Option 2: Use Python 3.11

Python 3.11 also works well with TensorFlow:
```bash
py -3.11 -m venv venv
venv\Scripts\activate
```

## Installation Steps

### 1. Create Virtual Environment (Recommended)

```bash
# Windows
py -3.12 -m venv venv
venv\Scripts\activate

# Mac/Linux
python3.12 -m venv venv
source venv/bin/activate
```

### 2. Install Dependencies

```bash
# Navigate to project directory
cd stock-prediction-lstm

# Install all dependencies
pip install -r requirements.txt
```

### 3. Verify Installation

```bash
python -c "import tensorflow as tf; print(f'TensorFlow version: {tf.__version__}')"
python -c "import yfinance; print('yfinance installed successfully')"
```

## Troubleshooting

### Issue: "No matching distribution found for tensorflow"

**Solution**: You're using Python 3.13+. Install Python 3.12 or 3.11.

### Issue: "ta-lib installation fails"

**Solution**: The project uses `ta` (not `ta-lib`), which doesn't require C libraries. Make sure you're installing from `requirements.txt`, not manually.

### Issue: "pip install fails with permission errors"

**Solution**: Use a virtual environment (see Step 1 above) or install with `--user` flag:
```bash
pip install --user -r requirements.txt
```

### Issue: TensorFlow installation is slow

**Solution**: This is normal. TensorFlow is a large package (~500MB). Be patient or use a faster internet connection.

## Alternative: Using Conda

If you prefer Conda:

```bash
# Create environment with Python 3.12
conda create -n stock-lstm python=3.12
conda activate stock-lstm

# Install dependencies
pip install -r requirements.txt
```

## Quick Test

After installation, test the setup:

```bash
python -c "from src.data_downloader import StockDataDownloader; print('Setup successful!')"
```

If you see "Setup successful!", you're ready to go!

