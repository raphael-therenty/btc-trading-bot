import os
import json
from datetime import datetime
import pandas as pd
import numpy as np

def _save_json(obj, name):
    os.makedirs("logs", exist_ok=True)
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    path = os.path.join("logs", f"{name}_{ts}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, default=str, indent=2)
    return path

def _max_drawdown(equity):
    peak = equity.cummax()
    dd = (equity - peak) / peak
    return float(dd.min())

def backtest_strategy(config, signals):
    """
    Backtest a naive long-only strategy:
      - signal_long = (xgb_prob > threshold) & (dnn_prob > threshold)
      - enter when signal goes 0->1, exit when 1->0
    Returns (trades_df, metrics)
    Saves trades CSV to outputs_models/trades.csv (or config['paths']['backtest_csv'] if provided)
    and metrics JSON to logs/.
    """
    paths = config.get("paths", {})
    processed_dir = paths.get("data_processed")
    processed_path = os.path.join(processed_dir, "crypto_processed.csv") if processed_dir else "data/processed/crypto_processed.csv"

    # Ensure signals is DataFrame with datetime index
    if not isinstance(signals, pd.DataFrame):
        signals = pd.DataFrame(signals)
    if signals.index.name != "datetime":
        try:
            signals.index = pd.to_datetime(signals.index)
            signals.index.name = "datetime"
        except Exception:
            pass

    # Load returns if not present in signals
    if 'BTC_ret' in signals.columns:
        returns = signals['BTC_ret'].astype(float)
    else:
        if not os.path.exists(processed_path):
            raise FileNotFoundError(f"Processed data not found at {processed_path}")
        proc = pd.read_csv(processed_path, parse_dates=['datetime'], index_col='datetime')
        if 'BTC_ret' in proc.columns:
            returns = proc['BTC_ret'].astype(float)
        else:
            close_cols = [c for c in proc.columns if 'close' in c.lower()]
            if not close_cols:
                raise ValueError("No returns or close column available for backtest.")
            returns = np.log(proc[close_cols[0]].astype(float)).diff()

    # align signals and returns on index intersection
    df = signals.join(returns.rename('BTC_ret'), how='inner')
    df = df.sort_index()
    if df.empty:
        raise ValueError("No overlapping dates between signals and returns for backtest.")

    # threshold from config (fallback 0.5)
    threshold = float(config.get('signals', {}).get('threshold', 0.5))

    xgb_p = df.get('xgb_prob', pd.Series(0.0, index=df.index)).fillna(0.0)
    dnn_p = df.get('dnn_prob', pd.Series(0.0, index=df.index)).fillna(0.0)

    df['position'] = ((xgb_p > threshold) & (dnn_p > threshold)).astype(int)

    # Strategy uses position shifted by 1 (enter next day)
    df['pos_lag'] = df['position'].shift(1).fillna(0).astype(int)
    df['strategy_ret'] = df['pos_lag'] * df['BTC_ret'].fillna(0.0)

    # equity curve (from log returns)
    df['cum_log'] = df['strategy_ret'].cumsum().fillna(0.0)
    df['equity'] = np.exp(df['cum_log'])

    # detect trades
    df['pos_diff'] = df['position'].diff().fillna(0)
    entries = df.index[df['pos_diff'] == 1].tolist()
    exits = df.index[df['pos_diff'] == -1].tolist()

    trades = []
    for entry in entries:
        # find corresponding exit after entry
        exit_candidates = [d for d in exits if d > entry]
        exit_date = exit_candidates[0] if exit_candidates else df.index[-1]
        # Compute realized log-return for the holding period:
        try:
            i_entry = df.index.get_loc(entry)
        except KeyError:
            continue
        start_i = i_entry + 1
        try:
            i_exit = df.index.get_loc(exit_date)
        except KeyError:
            i_exit = len(df) - 1
        if start_i > i_exit:
            realized_log = 0.0
            holding_days = 0
        else:
            realized_log = df['BTC_ret'].iloc[start_i:i_exit+1].sum()
            holding_days = i_exit - i_entry
        trades.append({
            "entry_date": str(entry.date()),
            "exit_date": str(exit_date.date()) if isinstance(exit_date, pd.Timestamp) else str(exit_date),
            "holding_days": int(holding_days),
            "realized_log_return": float(realized_log),
            "realized_pct_return": float(np.expm1(realized_log))
        })

    trades_df = pd.DataFrame(trades)

    # basic metrics
    returns_list = trades_df['realized_log_return'].values if not trades_df.empty else np.array([])
    n_trades = len(trades_df)
    n_realized = int(np.isfinite(returns_list).sum()) if returns_list.size else 0
    wins = int((returns_list > 0).sum()) if returns_list.size else 0
    losses = int((returns_list <= 0).sum()) if returns_list.size else 0
    win_rate = float(wins / n_realized) if n_realized > 0 else None
    avg_log = float(np.nanmean(returns_list)) if returns_list.size else None
    cum_log = float(np.nansum(returns_list)) if returns_list.size else 0.0
    vol = float(np.nanstd(returns_list, ddof=1)) if returns_list.size > 1 else None

    # annualized performance from equity curve
    daily_mean = float(df['strategy_ret'].mean()) if len(df) > 0 else 0.0
    daily_std = float(df['strategy_ret'].std(ddof=1)) if len(df) > 1 else 0.0
    ann_return = (np.exp(df['cum_log'].iloc[-1]) - 1) if not df['cum_log'].empty else 0.0
    ann_return_rate = (np.exp(daily_mean * 252) - 1) if daily_mean != 0.0 else 0.0
    ann_vol = daily_std * np.sqrt(252) if daily_std != 0.0 else None
    sharpe = (daily_mean / daily_std) * np.sqrt(252) if (daily_std not in (0.0, None)) else None

    max_dd = _max_drawdown(df['equity']) if not df['equity'].empty else None
    avg_holding = float(trades_df['holding_days'].mean()) if not trades_df.empty else None

    metrics = {
        "n_trades": n_trades,
        "n_realized": n_realized,
        "wins": wins,
        "losses": losses,
        "win_rate": win_rate,
        "avg_log_return_per_trade": avg_log,
        "cum_log_return_trades": cum_log,
        "strategy_annual_return_from_daily_mean": ann_return_rate,
        "strategy_annual_return_equity": ann_return,
        "strategy_annual_vol": ann_vol,
        "strategy_sharpe": sharpe,
        "max_drawdown": max_dd,
        "avg_holding_days": avg_holding,
        "threshold": threshold,
        "n_signal_days": int(df['position'].sum()),
        "data_start": str(df.index[0].date()),
        "data_end": str(df.index[-1].date())
    }

    # save trades and metrics
    out_dir = paths.get('outputs_models', 'data/outputs')
    os.makedirs(out_dir, exist_ok=True)
    trades_path = os.path.join(out_dir, "trades.csv")
    trades_df.to_csv(trades_path, index=False)
    metrics_path = _save_json(metrics, "backtest_metrics")

    return trades_df, metrics