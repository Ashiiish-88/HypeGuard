# HypeGuard 🚀🛡️

HypeGuard is an analytical pipeline designed to detect stock market anomalies, manipulated pumps, and "hype" driven by news or social sentiment. By leveraging financial market data (OHLCV) and real-time news headlines, HypeGuard engineers machine-learning-ready features to confidently separate organic market growth from artificial spikes.

## Features & Capabilities

- **Volume Anomaly Detection:** Calculates Relative Volume (RVOL), volume z-scores, and tracks persistent spikes.
- **Price Action Analysis:** Computes standard technical indicators like RSI (Relative Strength Index) and Bollinger Bands to identify overbought conditions and volatility expansion.
- **News & Sentiment Buzz:** Analyzes news density, extreme language ratios, source diversity, and headline similarities to identify coordinated spam or genuine catalysts.
- **Rule-based Hype Scoring:** Combines volume, price, and news signals (while accounting for real catalysts like earnings) to output a raw Hype Score and pseudo-labels for downstream Machine Learning models.

## Project Structure

```text
HypeGuard/
├── backend/
│   └── src/
│       ├── features.py          # Core feature engineering logic (Volume, Price, News)
│       └── scraper.py           # Data collection (yfinance, RSS/Web scraping)
├── data/                        # Output directory for generated charts and datasets
├── frontend/                    # Future frontend UI
├── notebooks/
│   └── 01_EDA_and_Features.ipynb # Presentation/Demo notebook for EDA and feature visuals
└── requirements.txt             # Python dependencies
```

## Setup & Installation

### 1. Prerequisites
Ensure you have Python 3.9+ installed on your system.

### 2. Create a Virtual Environment
It is highly recommended to use a virtual environment to keep dependencies isolated.
```bash
# Navigate to the project directory
cd HypeGuard

# Create a virtual environment named 'venv'
python -m venv venv

# Activate the virtual environment
# On Windows:
.\venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### 3. Install Dependencies
With the virtual environment activated, install the required packages.

Your `requirements.txt` should contain the following core libraries used in this project:
```txt
# ── Core Data ──────────────────────────────────────────────────────────
yfinance>=0.2.40          # Yahoo Finance OHLCV data (no API key needed)
feedparser>=6.0.11        # Google News RSS scraping (no API key needed)
pandas>=2.0.0
numpy>=1.26.0

# ── ML / Feature Engineering ───────────────────────────────────────────
scikit-learn>=1.4.0       # Isolation Forest, Random Forest, preprocessing

# ── Notebook / EDA ─────────────────────────────────────────────────────
jupyter>=1.0.0
matplotlib>=3.8.0
seaborn>=0.13.0
```

Install them by running:
```bash
pip install -r requirements.txt
```

### 4. Jupyter Notebook Setup (VS Code)
To run the analysis and presentation notebooks correctly:
1. Open `notebooks/01_EDA_and_Features.ipynb` in VS Code.
2. In the top right corner of the notebook, click on the **Select Kernel** button.
3. Select **Python Environments**.
4. Choose the `venv` environment you just created (e.g., `venv (Python 3.x.x)` or `d:\HypeGuard\venv\Scripts\python.exe`).
5. (Optional) If prompted, allow VS Code to install the `ipykernel` package into the virtual environment.

## Usage

Start by exploring the core EDA (Exploratory Data Analysis) notebook:
- Run `notebooks/01_EDA_and_Features.ipynb` from top to bottom.
- The notebook will pull data for sample tickers (like GME, NVDA, AAPL), run them through the `features.py` pipeline, and output visual charts showing volume spikes, RSI thresholds, and final Hype Scores into the `data/` folder.
