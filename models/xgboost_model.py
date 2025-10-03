import os
import json
import warnings
from datetime import datetime
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import roc_auc_score

def _save_log(log: dict, name: str):
    os.makedirs("logs", exist_ok=True)
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    path = os.path.join("logs", f"{name}_{ts}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(log, f, default=str, indent=2)
    return path

def train_xgb(config):
    paths = config.get('paths', {})
    proc_dir = paths.get('data_processed')
    proc_path = os.path.join(proc_dir, "crypto_processed.csv")
    if not os.path.exists(proc_path):
        raise FileNotFoundError(proc_path)

    df = pd.read_csv(proc_path, parse_dates=['datetime'], index_col='datetime')

    # training window from config
    train_cfg = config.get('training', {})
    start = pd.to_datetime(train_cfg.get('train_start')) if train_cfg.get('train_start') else df.index[0]
    end = pd.to_datetime(train_cfg.get('train_end')) if train_cfg.get('train_end') else df.index[-1]
    df_window = df.loc[(df.index >= start) & (df.index <= end)].copy()
    test_size = float(train_cfg.get('test_size', 0.2))

    # features selection
    feature_cols = [c for c in df_window.columns if c.upper().startswith('BTC') and 'ret' not in c.lower()]
    if not feature_cols:
        feature_cols = [c for c in df_window.select_dtypes(include=[np.number]).columns if 'ret' not in c.lower()]

    # target
    if 'BTC_ret' in df_window.columns:
        y = (df_window['BTC_ret'].shift(-1) > 0).astype(int)
    else:
        close_cols = [c for c in df_window.columns if 'close' in c.lower()]
        if not close_cols:
            raise ValueError("No return or close column found for training XGB")
        y = (np.log(df_window[close_cols[0]]).diff().shift(-1) > 0).astype(int)

    # align and drop rows with no target
    data_idx = y.dropna().index
    X = df_window.loc[data_idx, feature_cols].astype(float).fillna(0.0)
    y = y.loc[data_idx].astype(int)

    n = len(X)
    split = max(1, int(n * (1 - test_size)))
    print(f"XGB training window: {start.date()} -> {end.date()}  rows={n}; train=0..{split-1}, val={split}..{n-1} (test_size={test_size})")

    X_train, X_val = X.iloc[:split], X.iloc[split:]
    y_train, y_val = y.iloc[:split], y.iloc[split:]

    params = config.get('training', {}).get('xgb', {})
    model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss',
                              max_depth=int(params.get('max_depth', 3)),
                              n_estimators=int(params.get('n_estimators', 100)))
    model.fit(X_train, y_train)

    auc = None
    try:
        preds = model.predict_proba(X_val)[:, 1]
        auc = float(roc_auc_score(y_val, preds))
        print(f"XGB validation AUC: {auc:.4f}")
    except Exception:
        pass

    # save run log
    log = {
        "model": "xgboost",
        "params": params,
        "train_start": str(start.date()),
        "train_end": str(end.date()),
        "n_total": n,
        "n_train": len(X_train),
        "n_val": len(X_val),
        "val_auc": auc,
        "feature_cols": feature_cols
    }
    log_path = _save_log(log, "train_xgb")
    print(f"Saved XGB run log to {log_path}")

    # attach feature names for prediction alignment
    try:
        model.get_booster().feature_names = feature_cols
    except Exception:
        setattr(model, "feature_names_in_", feature_cols)

    return model


def predict_xgb(config, model):
    paths = config.get('paths', {})
    proc_dir = paths.get('data_processed')
    proc_path = os.path.join(proc_dir, "crypto_processed.csv")
    if not os.path.exists(proc_path):
        raise FileNotFoundError(proc_path)

    df = pd.read_csv(proc_path, parse_dates=['datetime'], index_col='datetime')

    # expected features
    expected = None
    try:
        booster = model.get_booster()
        expected = booster.feature_names
    except Exception:
        expected = getattr(model, "feature_names_in_", None)
    if expected is None:
        expected = [c for c in df.columns if c.upper().startswith('BTC') and 'ret' not in c.lower()]
        if not expected:
            expected = [c for c in df.select_dtypes(include=[np.number]).columns if 'ret' not in c.lower()]

    # align columns and fill missing
    X = pd.DataFrame(index=df.index)
    for col in expected:
        X[col] = pd.to_numeric(df[col], errors='coerce') if col in df.columns else 0.0
    X = X.astype(float).fillna(0.0)

    try:
        probs = model.predict_proba(X)[:, 1]
    except Exception:
        # fallback to raw booster
        try:
            booster = model if isinstance(model, xgb.Booster) else model.get_booster()
            dmat = xgb.DMatrix(X.values, feature_names=expected)
            probs = booster.predict(dmat)
            if probs.ndim == 2 and probs.shape[1] == 2:
                probs = probs[:, 1]
        except Exception as e:
            raise RuntimeError(f"Failed to produce XGB predictions: {e}")

    return pd.Series(probs, index=X.index, name='xgb_prob')