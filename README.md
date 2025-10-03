# Bitcoin Trading Bot

A Python-based project to download historical BTC data, process features, train predictive models, generate trading signals, and backtest a simple trading strategy.

---

## Table of Contents

1. [Project Overview](#project-overview)  
2. [Project Structure](#project-structure)  
3. [Dependencies](#dependencies)  
4. [Configuration](#configuration)  
5. [Setup Instructions](#setup-instructions)  
6. [Running the Pipeline](#running-the-pipeline)  
7. [Understanding the Workflow](#understanding-the-workflow)  
8. [Output Files](#output-files)  
9. [Testing](#testing)  
10. [Extending the Project](#extending-the-project)

---

## Project Overview

This project implements a BTC trading bot with the following capabilities:

- **Data Acquisition**  
  - Downloads historical BTC/USDT data from Binance.  
  - Optionally downloads US macroeconomic indicators (FRED API).

- **Feature Engineering**  
  - Computes technical indicators (returns, momentum, volatility, ATR, ADX, EMA spreads, breakout signals).  
  - Merges macroeconomic and optional on-chain data.

- **Modeling**  
  - **XGBoost**: Predicts the probability of a positive return.  
  - **Deep Neural Network (DNN)**: Same objective, learns non-linear patterns.  
  - **GARCH**: Forecasts volatility for risk-adjusted strategy.

- **Signal Generation & Backtesting**  
  - Combines model probabilities into a trading signal.  
  - Simple backtest evaluates cumulative returns based on position and forecasted volatility.

---

## Project Structure

```
btc_trading_bot/
│── data/                  
│   ├── raw/               # Raw Binance + macro CSVs
│   └── processed/         # Feature-engineered CSVs
│── outputs/
│   ├── models/            # Model predictions
│   └── backtests/         # Backtest metrics & plots
│── data_scripts/
│   ├── download_binance.py
│   ├── download_macro.py
│   └── preprocess_data.py
│── models/
│   ├── xgboost_model.py
│   ├── dnn_model.py
│   └── garch_model.py
│── utils/
│   ├── features.py
│   ├── backtest.py
│   └── config_loader.py
│── tests/
│   ├── test_data.py
│   ├── test_models.py
│   └── test_backtest.py
│── config.yaml            # Central config (hyperparameters & paths)
│── main.py                # Orchestrator script
│── requirements.txt
│── README.md
│── .gitignore
```

---

## Dependencies

Install the required packages via `pip`:

```bash
pip install -r requirements.txt
```

Key libraries:

- `pandas`, `numpy` → Data manipulation.  
- `python-binance` → Download BTC data.  
- `pandas-datareader` → Access macro data (FRED).  
- `xgboost` → Gradient boosting classifier.  
- `tensorflow` → Neural network modeling.  
- `arch` → Volatility forecasting with GARCH.  
- `ta` → Technical indicators (ADX, ATR, momentum, etc.).  
- `PyYAML` → Load configuration files.  
- `pytest` → Unit testing.

---

## Configuration

All parameters are centralized in `config.yaml`:

- **Paths**: raw data, processed data, model outputs, backtest outputs.  
- **Strategy**: probability thresholds, daily volatility target, max leverage.  
- **Models**: hyperparameters for XGBoost, DNN, and GARCH.  
- **APIs**: Binance keys, FRED API key.  

> Make sure to update API keys in `config.yaml` if you want to use private data endpoints.

---

## Setup Instructions

1. **Clone the repository**:

```bash
git clone <your-repo-url>
cd btc_trading_bot
```

2. **Create Python virtual environment** (recommended):

```bash
python -m venv venv
source venv/bin/activate  # macOS/Linux
Then
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process
.\venv\Scripts\activate     # Windows
```

3. **Install dependencies**:

```bash
pip install -r requirements.txt
```

4. **Check config.yaml**:  
   - Set Binance API keys if you have them.  
   - Set FRED API key if macro data is desired.

---

## Running the Pipeline

Once everything is installed:

```bash
python main.py
```

This will:

1. Download historical BTC/USDT data (from Binance) and macro data (optional).  
2. Preprocess raw data into features.  
3. Train the XGBoost, DNN, and GARCH models.  
4. Generate probability predictions and volatility forecasts.  
5. Save predictions and signals to `outputs/models/`.  
6. Run a backtest and save results to `outputs/backtests/`.

---

## Understanding the Workflow

1. **Download Data** (`data_scripts/`):
   - `download_binance.py` → BTC prices.  
   - `download_macro.py` → US macroeconomic indicators (CPI, unemployment, GDP).  

2. **Preprocessing** (`preprocess_data.py` & `features.py`):
   - Compute technical features.  
   - Merge macro and optional on-chain data.  
   - Output is a clean, processed CSV.

3. **Model Training** (`models/`):
   - `xgboost_model.py` → Binary classification on BTC return direction.  
   - `dnn_model.py` → Neural network for same task.  
   - `garch_model.py` → Forecast BTC volatility.

4. **Signals & Backtest** (`utils/backtest.py`):
   - Generate long/short signals based on probability thresholds.  
   - Compute daily returns and cumulative P&L.

---

## Output Files

- **Model Predictions** (`outputs/models/`):
  - `BTC_xgb_prob.csv` → XGBoost probabilities.  
  - `BTC_dnn_prob.csv` → DNN probabilities.  
  - `BTC_garch_forecast.csv` → Forecasted volatility.  
  - `signals.csv` → Combined trading signals.

- **Backtest Results** (`outputs/backtests/`):
  - `backtest_enhanced.csv` → Positions, daily returns, cumulative P&L.

---

## Testing

Run unit tests to verify each module:

```bash
pytest tests/
```

Tests include:

- Data download & preprocessing.  
- Model training & prediction.  
- Backtest execution.

---

## Extending the Project

- Add more technical indicators or macro variables in `features.py`.  
- Include more complex models (LSTM, CNN) in `models/`.  
- Implement more sophisticated backtesting or portfolio management in `backtest.py`.  
- Add plotting and analysis notebooks in `notebooks/`.

---

## Notes

- This project is for **educational purposes** and does **not constitute financial advice**.  
- Always test thoroughly before attempting live trading.  
- Ensure you have API keys and internet access for real-time data fetching.