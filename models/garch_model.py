import os
import warnings
import pandas as pd
import numpy as np

# Try to use arch if available, otherwise fallback to simple rolling-volatility
try:
    from arch import arch_model  # type: ignore
except Exception:
    arch_model = None


def forecast_garch(config, returns=None, horizon=1):
    """
    Forecast 1-day volatility.
    - If `returns` (pd.Series) is provided, use it.
    - Otherwise load processed data and infer returns column (BTC_ret or similar).
    - If arch is available, fit GARCH(1,1). Otherwise use rolling std as fallback.
    Returns pd.Series of volatility (decimal, not percentage) indexed like input returns (last value(s) for horizon).
    """
    paths = config.get('paths', {})

    # Load returns if not provided
    if returns is None:
        processed_dir = paths.get('data_processed')
        if processed_dir is None:
            raise KeyError("config['paths']['data_processed'] is required")
        processed_path = os.path.join(processed_dir, "crypto_processed.csv")
        if not os.path.exists(processed_path):
            raise FileNotFoundError(f"Processed data not found at {processed_path!r}")
        df = pd.read_csv(processed_path, parse_dates=['datetime'], index_col='datetime')
        rcols = [c for c in df.columns if 'ret' in c.lower() or 'return' in c.lower()]
        if not rcols:
            close_candidates = [c for c in df.columns if 'close' in c.lower()]
            if not close_candidates:
                raise ValueError("No returns or close column found in processed data to compute volatility.")
            close = df[close_candidates[0]].astype(float)
            returns = np.log(close).diff().dropna()
        else:
            returns = df[rcols[0]].astype(float).dropna()

    # Ensure returns is a pd.Series
    if isinstance(returns, pd.DataFrame):
        returns = returns.iloc[:, 0]
    returns = returns.dropna()
    if returns.empty:
        raise ValueError("Returns series is empty - cannot forecast volatility.")

    # Use ARCH if available
    if arch_model is not None:
        try:
            am = arch_model(returns * 100, vol='Garch', p=1, q=1, dist='normal')
            res = am.fit(disp='off', show_warning=False)
            f = res.forecast(horizon=horizon, reindex=False)
            var_forecast = f.variance.iloc[-1, 0] / (100.0 ** 2)
            last_idx = returns.index[-1]
            return pd.Series([np.sqrt(var_forecast)], index=[last_idx], name='garch_vol')
        except Exception as e:
            warnings.warn(f"ARCH model fitting failed: {e}. Falling back to rolling-volatility.")

    # Fallback: rolling std
    window = min(63, max(5, int(len(returns) / 10)))
    roll_std = returns.rolling(window=window).std().dropna()
    if roll_std.empty:
        vol = returns.std()
    else:
        vol = roll_std.iloc[-1]
    last_idx = returns.index[-1] if hasattr(returns, 'index') else 0
    return pd.Series([vol], index=[last_idx], name='garch_vol')
