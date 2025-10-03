import os
import json
from datetime import datetime
import yaml
import pandas as pd
import numpy as np

cfg = yaml.safe_load(open("config.yaml"))
signals_path = cfg['paths'].get('signals_csv')
proc_path = os.path.join(cfg['paths']['data_processed'], "crypto_processed.csv")

if not os.path.exists(signals_path):
    raise FileNotFoundError(f"Signals CSV not found: {signals_path}")
if not os.path.exists(proc_path):
    raise FileNotFoundError(f"Processed CSV not found: {proc_path}")

signals = pd.read_csv(signals_path, parse_dates=['datetime'], index_col='datetime')
proc = pd.read_csv(proc_path, parse_dates=['datetime'], index_col='datetime')

# Align and compute next-day returns if available
# Prefer BTC_ret if present; else compute from close
if 'BTC_ret' in proc.columns:
    returns = proc['BTC_ret']
else:
    close_cols = [c for c in proc.columns if 'close' in c.lower()]
    if not close_cols:
        raise ValueError("No return or close column available to compute performance.")
    returns = np.log(proc[close_cols[0]]).diff()

# Align signals and returns
df = signals.join(returns.rename('BTC_ret'), how='inner')
df = df.dropna(subset=['BTC_ret'])
if df.empty:
    raise ValueError("No overlapping dates between signals and returns.")

# Define simple rule: long when both models predict > threshold
threshold = 0.5
df['signal_long'] = ((df.get('xgb_prob', 0) > threshold) & (df.get('dnn_prob', 0) > threshold)).astype(int)

# Identify trade entries (signal changes from 0->1)
df['entry'] = (df['signal_long'].diff() == 1)
df['exit'] = (df['signal_long'].diff() == -1)

# For realized trades compute the return from entry date to next day (use BTC_ret shifted -1)
# We'll consider trades that have BTC_ret available on the day after entry
entries = df[df['entry']]
n_entries = len(entries)
realized = 0
wins = 0
losses = 0
returns_list = []

for dt, row in entries.iterrows():
    # next-day return available at dt (since BTC_ret is defined as log-return from t-1->t in project)
    # Use BTC_ret.shift(-1) to represent return from entry day to next day if needed
    try:
        # If BTC_ret here is return from previous day to this day, the realized next-day return is at next index
        next_idx = df.index.get_indexer([dt + pd.Timedelta(days=1)])[0]
        if next_idx == -1:
            continue
        r = df['BTC_ret'].iloc[next_idx]
    except Exception:
        # fallback: use the next row's BTC_ret if exists
        pos = df.index.get_loc(dt)
        if pos + 1 >= len(df):
            continue
        r = df['BTC_ret'].iloc[pos + 1]
    realized += 1
    returns_list.append(r)
    if r > 0:
        wins += 1
    else:
        losses += 1

# Performance metrics
cum_return = np.nansum(returns_list)
n_realized = realized
win_rate = wins / realized if realized > 0 else None
avg_return = np.nanmean(returns_list) if returns_list else None
vol = np.nanstd(returns_list, ddof=1) if len(returns_list) > 1 else None
sharpe = (avg_return / vol) * np.sqrt(252) if (avg_return is not None and vol not in (None, 0)) else None

summary = {
    "n_signals_total": len(df),
    "n_entries": n_entries,
    "n_realized_trades": n_realized,
    "wins": wins,
    "losses": losses,
    "win_rate": win_rate,
    "cum_return_log": cum_return,
    "avg_return_per_trade_log": avg_return,
    "volatility_per_trade_log": vol,
    "sharpe_annualized": sharpe,
    "threshold": threshold,
    "generated_at": datetime.utcnow().isoformat()
}

os.makedirs("logs", exist_ok=True)
out = os.path.join("logs", f"backtest_summary_{datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')}.json")
with open(out, "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2, default=str)

print("Backtest / signals summary:")
print(json.dumps(summary, indent=2, default=str))
print(f"Saved summary to {out}")