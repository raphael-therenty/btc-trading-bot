import pandas as pd
import numpy as np

def compute_technical_indicators(df_crypto: pd.DataFrame) -> pd.DataFrame:
    """
    Build a small set of crypto features from raw OHLCV-like data.
    - Detects a close column and volume column heuristically.
    - Returns a DataFrame indexed by the same datetime index with columns
      that start with 'BTC' so downstream code can detect them.
    """
    if df_crypto is None or df_crypto.empty:
        return pd.DataFrame(index=df_crypto.index if df_crypto is not None else [])

    df = df_crypto.copy()

    # heuristics for close and volume column names
    lower_cols = {c: c.lower() for c in df.columns}
    close_col = None
    for cand in ["close", "adjclose", "adj_close", "price", "last", "close_usd"]:
        for c, lc in lower_cols.items():
            if cand in lc:
                close_col = c
                break
        if close_col:
            break
    if close_col is None:
        # fallback: first numeric column
        numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        if numeric_cols:
            close_col = numeric_cols[0]
        else:
            # nothing to do
            return pd.DataFrame(index=df.index)

    vol_col = None
    for cand in ["volume", "vol"]:
        for c, lc in lower_cols.items():
            if cand in lc:
                vol_col = c
                break
        if vol_col:
            break

    # ensure numeric
    close = pd.to_numeric(df[close_col], errors="coerce").astype(float).copy()
    close.name = "BTC_close"
    if vol_col:
        volume = pd.to_numeric(df[vol_col], errors="coerce").astype(float).copy()
        volume.name = "BTC_volume"
    else:
        volume = None

    # compute log price and returns
    btc_logp = np.log(close).rename("BTC_logp")
    btc_ret = btc_logp.diff().rename("BTC_ret")

    # moving averages
    btc_sma10 = close.rolling(window=10, min_periods=1).mean().rename("BTC_sma10")
    btc_sma50 = close.rolling(window=50, min_periods=1).mean().rename("BTC_sma50")

    # RSI (14)
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=14, min_periods=14).mean()
    avg_loss = loss.rolling(window=14, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = (100 - (100 / (1 + rs))).rename("BTC_rsi14")
    # first RSI values will be NaN; that's OK

    pieces = [btc_logp, btc_ret, btc_sma10, btc_sma50, rsi]
    if volume is not None:
        pieces.insert(0, volume.rename("BTC_volume"))
    # also include raw close as BTC_close
    pieces.insert(0, close.rename("BTC_close"))

    df_features = pd.concat(pieces, axis=1)
    df_features.index.name = df.index.name if df.index.name else "datetime"
    return df_features


def merge_macro_onchain(df_features: pd.DataFrame, df_macro: pd.DataFrame, df_onchain: pd.DataFrame) -> pd.DataFrame:
    """
    Merge feature dataframe with macro and onchain data.
    - Aligns on the index (assumed datetime-like).
    - Keeps all feature rows (features are the driver), allows macro/onchain to be NaN.
    - Returns a single DataFrame with columns from all sources.
    """
    # Ensure we have empty DataFrames instead of None
    pieces = []
    if df_features is None:
        df_features = pd.DataFrame()
    if df_macro is None:
        df_macro = pd.DataFrame()
    if df_onchain is None:
        df_onchain = pd.DataFrame()

    # Make sure indexes are datetime-like and named consistently
    for d in (df_features, df_macro, df_onchain):
        if d is not None and not d.empty:
            try:
                d.index = pd.to_datetime(d.index, errors="coerce")
            except Exception:
                pass

    # Prefer features' index as the base index to preserve crypto rows
    base_index = None
    if not df_features.empty:
        base_index = df_features.index
    else:
        # fallback to union of others
        idxs = []
        if not df_macro.empty:
            idxs.append(df_macro.index)
        if not df_onchain.empty:
            idxs.append(df_onchain.index)
        if idxs:
            base_index = idxs[0].union_many(idxs[1:]) if len(idxs) > 1 else idxs[0]
        else:
            base_index = pd.DatetimeIndex([])

    # concat via outer join to include macro/onchain columns; then reindex to base_index to keep feature rows
    merged = pd.concat([df_features, df_macro, df_onchain], axis=1, join="outer", copy=False)
    if not base_index.empty:
        merged = merged.reindex(base_index)
    merged = merged.sort_index()
    if merged.index.name is None:
        merged.index.name = "datetime"
    return merged