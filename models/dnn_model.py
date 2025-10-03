import os
import json
from datetime import datetime
import pandas as pd
import numpy as np
from tensorflow import keras

def _save_log(log: dict, name: str):
    os.makedirs("logs", exist_ok=True)
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    path = os.path.join("logs", f"{name}_{ts}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(log, f, default=str, indent=2)
    return path

def train_dnn(config):
    paths = config.get('paths', {})
    proc_dir = paths.get('data_processed')
    proc_path = os.path.join(proc_dir, "crypto_processed.csv")
    if not os.path.exists(proc_path):
        raise FileNotFoundError(proc_path)

    df = pd.read_csv(proc_path, parse_dates=['datetime'], index_col='datetime')

    train_cfg = config.get('training', {})
    start = pd.to_datetime(train_cfg.get('train_start')) if train_cfg.get('train_start') else df.index[0]
    end = pd.to_datetime(train_cfg.get('train_end')) if train_cfg.get('train_end') else df.index[-1]
    df_window = df.loc[(df.index >= start) & (df.index <= end)].copy()
    test_size = float(train_cfg.get('test_size', 0.2))

    feature_cols = [c for c in df_window.columns if c.upper().startswith('BTC') and 'ret' not in c.lower()]
    if not feature_cols:
        feature_cols = [c for c in df_window.select_dtypes(include=[np.number]).columns if 'ret' not in c.lower()]

    if 'BTC_ret' in df_window.columns:
        y = (df_window['BTC_ret'].shift(-1) > 0).astype(int)
    else:
        close_cols = [c for c in df_window.columns if 'close' in c.lower()]
        if not close_cols:
            raise ValueError("No return or close column found for DNN training")
        y = (np.log(df_window[close_cols[0]]).diff().shift(-1) > 0).astype(int)

    data_idx = y.dropna().index
    X = df_window.loc[data_idx, feature_cols].astype(float).fillna(0.0)
    y = y.loc[data_idx].astype(int)

    n = len(X)
    split = max(1, int(n * (1 - test_size)))
    print(f"DNN training window: {start.date()} -> {end.date()}  rows={n}; train=0..{split-1}, val={split}..{n-1} (test_size={test_size})")

    X_train, X_val = X.iloc[:split], X.iloc[split:]
    y_train, y_val = y.iloc[:split], y.iloc[split:]

    dnn_cfg = config.get('training', {}).get('dnn', {})
    epochs = int(dnn_cfg.get('epochs', 5))
    batch_size = int(dnn_cfg.get('batch_size', 32))

    model = keras.Sequential([
        keras.Input(shape=(X.shape[1],)),
        keras.layers.Dense(32, activation='relu'),
        keras.layers.Dense(16, activation='relu'),
        keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    history = model.fit(X_train.values, y_train.values, epochs=epochs, batch_size=batch_size, verbose=0, validation_data=(X_val.values, y_val.values))

    # compute final val accuracy
    val_acc = None
    if 'val_accuracy' in history.history:
        val_acc = float(history.history['val_accuracy'][-1]) if history.history['val_accuracy'] else None

    # attach feature names
    model._feature_names = feature_cols

    # save run log
    log = {
        "model": "dnn",
        "params": {"epochs": epochs, "batch_size": batch_size},
        "train_start": str(start.date()),
        "train_end": str(end.date()),
        "n_total": n,
        "n_train": len(X_train),
        "n_val": len(X_val),
        "val_accuracy": val_acc,
        "feature_cols": feature_cols
    }
    log_path = _save_log(log, "train_dnn")
    print(f"Saved DNN run log to {log_path}")

    return model


def predict_dnn(config, model):
    paths = config.get('paths', {})
    proc_dir = paths.get('data_processed')
    proc_path = os.path.join(proc_dir, "crypto_processed.csv")
    if not os.path.exists(proc_path):
        raise FileNotFoundError(proc_path)

    df = pd.read_csv(proc_path, parse_dates=['datetime'], index_col='datetime')

    expected = getattr(model, "_feature_names", None)
    if expected is None:
        expected = [c for c in df.columns if c.upper().startswith('BTC') and 'ret' not in c.lower()]
        if not expected:
            expected = [c for c in df.select_dtypes(include=[np.number]).columns if 'ret' not in c.lower()]

    X = pd.DataFrame(index=df.index)
    for col in expected:
        X[col] = pd.to_numeric(df[col], errors='coerce') if col in df.columns else 0.0
    X = X.astype(float).fillna(0.0)

    preds = model.predict(X.values).squeeze()
    if preds.shape[0] != len(X):
        preds = np.resize(preds, len(X))
    return pd.Series(preds, index=X.index, name='dnn_prob')