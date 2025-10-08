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

def backtest_strategy(config, signals, proc_df=None, backtest_window=None):
    """
    Backtest a naive long-only strategy on the provided signals DataFrame.
    - signals: DataFrame indexed by datetime with columns 'xgb_prob' and 'dnn_prob' (and optional 'garch_vol').
    - proc_df: processed dataframe (must contain BTC_close). If not provided, will be read from config paths.
    - backtest_window: (start, end) tuple of Timestamps for clarity (optional).
    Position sizing: uses config['trading'].initial_capital and position_fraction (fraction of capital used per trade).
    Entry is executed at next available close after the signal (enter next day).
    Returns (trades_df, metrics) and saves trades.csv + metrics json.
    """
    paths = config.get("paths", {})
    processed_dir = paths.get("data_processed")
    processed_path = os.path.join(processed_dir, "crypto_processed.csv") if processed_dir else "data/processed/crypto_processed.csv"

    if proc_df is None:
        if not os.path.exists(processed_path):
            raise FileNotFoundError(f"Processed data not found at {processed_path}")
        proc_df = pd.read_csv(processed_path, parse_dates=['datetime'], index_col='datetime')

    # ensure index and float prices present
    proc_df = proc_df.sort_index()
    if 'BTC_close' not in proc_df.columns:
        raise ValueError("Processed data must contain BTC_close column for price-based backtest.")

    # filter to backtest window if provided
    if backtest_window is not None:
        start, end = backtest_window
        signals = signals.loc[(signals.index >= start) & (signals.index <= end)]
        proc_df = proc_df.loc[(proc_df.index >= start) & (proc_df.index <= end)]

    # align signals and price
    df = signals.join(proc_df[['BTC_close', 'BTC_ret']], how='left')
    df = df.sort_index().dropna(subset=['BTC_close'])
    if df.empty:
        raise ValueError("No overlapping data between signals and processed prices for backtest.")

    # config for sizing
    trading_cfg = config.get('training', {})
    initial_capital = float(trading_cfg.get('initial_capital', 10000.0))
    position_fraction = float(trading_cfg.get('position_fraction', 1.0))  # fraction of capital used per trade

    # threshold
    threshold = float(config.get('signals', {}).get('threshold', 0.5))

    xgb_p = df.get('xgb_prob', pd.Series(0.0, index=df.index)).fillna(0.0)
    dnn_p = df.get('dnn_prob', pd.Series(0.0, index=df.index)).fillna(0.0)
    df['signal'] = ((xgb_p > threshold) & (dnn_p > threshold)).astype(int)

    # detect transitions on each date; we execute orders at next available close
    df['pos_diff'] = df['signal'].diff().fillna(0)
    entries = df.index[df['pos_diff'] == 1].tolist()
    exits = df.index[df['pos_diff'] == -1].tolist()

    trades = []
    cash = initial_capital
    in_position = False
    for entry in entries:
        # entry execution price = close at next row after entry (enter next day)
        try:
            i_entry = df.index.get_loc(entry)
        except KeyError:
            continue
        entry_exec_idx = i_entry + 1
        if entry_exec_idx >= len(df):
            # no next-day price => cannot open
            continue
        entry_price = float(df['BTC_close'].iloc[entry_exec_idx])
        # find exit date after entry
        exit_candidates = [d for d in exits if d > entry]
        exit_signal_date = exit_candidates[0] if exit_candidates else None
        if exit_signal_date is None:
            # close at last available close
            exit_exec_idx = len(df) - 1
        else:
            try:
                i_exit_signal = df.index.get_loc(exit_signal_date)
                exit_exec_idx = i_exit_signal + 1
                if exit_exec_idx >= len(df):
                    exit_exec_idx = len(df) - 1
            except KeyError:
                exit_exec_idx = len(df) - 1

        exit_price = float(df['BTC_close'].iloc[exit_exec_idx])

        # compute shares to buy (floor)
        alloc = initial_capital * position_fraction
        shares = int(np.floor(alloc / entry_price))
        if shares <= 0:
            # cannot buy fractional shares, skip trade
            continue

        pnl = shares * (exit_price - entry_price)
        pct_return = (exit_price / entry_price) - 1.0
        log_return = float(np.log(exit_price / entry_price)) if entry_price > 0 else 0.0
        holding_days = exit_exec_idx - entry_exec_idx + 1 if exit_exec_idx >= entry_exec_idx else 0

        trades.append({
            "entry_signal_date": str(entry.date()),
            "entry_exec_date": str(df.index[entry_exec_idx].date()),
            "entry_price": float(entry_price),
            "exit_signal_date": str(exit_signal_date.date()) if exit_signal_date is not None else None,
            "exit_exec_date": str(df.index[exit_exec_idx].date()),
            "exit_price": float(exit_price),
            "shares": int(shares),
            "pnl": float(pnl),
            "pct_return": float(pct_return),
            "log_return": float(log_return),
            "holding_days": int(holding_days)
        })

    trades_df = pd.DataFrame(trades)

    # compute metrics
    if trades_df.empty:
        metrics = {
            "n_trades": 0,
            "n_realized": 0,
            "wins": 0,
            "losses": 0,
            "win_rate": None,
            "cum_pnl": 0.0,
            "cum_return": 0.0,
            "avg_pnl": None,
            "avg_pct_return": None,
            "strategy_sharpe": None,
            "max_drawdown": 0.0,
            "avg_holding_days": None,
            "threshold": threshold,
            "data_start": str(df.index[0].date()),
            "data_end": str(df.index[-1].date())
        }
    else:
        pnl_list = trades_df['pnl'].values
        pct_list = trades_df['pct_return'].values
        n_trades = len(trades_df)
        wins = int((pnl_list > 0).sum())
        losses = int((pnl_list <= 0).sum())
        win_rate = float(wins / n_trades) if n_trades > 0 else None
        cum_pnl = float(np.nansum(pnl_list))
        # cumulative return relative to capital
        cum_return = float(cum_pnl / initial_capital)
        avg_pnl = float(np.nanmean(pnl_list))
        avg_pct = float(np.nanmean(pct_list))
        # approximate per-day strategy returns via equity curve built from daily strategy_ret
        # Build daily equity series
        df_daily = df.copy()
        df_daily['pos'] = df_daily['signal'].shift(1).fillna(0).astype(int)
        df_daily['daily_ret'] = df_daily['pos'] * df_daily.get('BTC_ret', pd.Series(0.0, index=df_daily.index)).fillna(0.0)
        cum_log = df_daily['daily_ret'].cumsum()
        equity = np.exp(cum_log)
        daily_mean = float(df_daily['daily_ret'].mean())
        daily_std = float(df_daily['daily_ret'].std(ddof=1)) if len(df_daily) > 1 else 0.0
        ann_return = (np.exp(cum_log.iloc[-1]) - 1) if not cum_log.empty else 0.0
        ann_return_rate = (np.exp(daily_mean * 252) - 1) if daily_mean != 0.0 else 0.0
        ann_vol = daily_std * np.sqrt(252) if daily_std != 0.0 else None
        sharpe = (daily_mean / daily_std) * np.sqrt(252) if (daily_std not in (0.0, None, 0.0)) else None
        max_dd = _max_drawdown(equity) if not equity.empty else None
        avg_holding = float(trades_df['holding_days'].mean())

        metrics = {
            "n_trades": int(n_trades),
            "n_realized": int(n_trades),
            "wins": int(wins),
            "losses": int(losses),
            "win_rate": win_rate,
            "cum_pnl": cum_pnl,
            "cum_return": cum_return,
            "avg_pnl": avg_pnl,
            "avg_pct_return": avg_pct,
            "strategy_annual_return_from_daily_mean": ann_return_rate,
            "strategy_annual_return_equity": ann_return,
            "strategy_annual_vol": ann_vol,
            "strategy_sharpe": sharpe,
            "max_drawdown": max_dd,
            "avg_holding_days": avg_holding,
            "threshold": threshold,
            "n_signal_days": int(df['signal'].sum()),
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