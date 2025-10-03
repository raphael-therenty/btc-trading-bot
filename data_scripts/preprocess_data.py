import os
import warnings
import pandas as pd
from utils.features import compute_technical_indicators, merge_macro_onchain

def preprocess_data(config):
    """
    Preprocess crypto, macro, and optional onchain data into a single CSV.
    Ensures crypto rows are preserved (macro/onchain may be NaN).
    """
    paths = config.get('paths', {})
    crypto_path = paths.get('crypto_csv')
    if not crypto_path or not os.path.exists(crypto_path):
        raise FileNotFoundError(f"Crypto CSV not found at {crypto_path!r}.")

    df_crypto = pd.read_csv(crypto_path, parse_dates=['datetime'], index_col='datetime').sort_index()
    if df_crypto.index.name is None:
        df_crypto.index.name = 'datetime'

    # Macro
    macro_path = paths.get('macro_csv')
    df_macro = pd.DataFrame()
    if macro_path and os.path.exists(macro_path):
        df_macro = pd.read_csv(macro_path, parse_dates=['Date'], index_col='Date')
        df_macro.index = pd.to_datetime(df_macro.index, errors='coerce')
        df_macro = df_macro[~df_macro.index.isna()]
        if df_macro.index.has_duplicates:
            df_macro = df_macro.groupby(df_macro.index).last()
        start = min(df_macro.index.min(), df_crypto.index.min())
        end = max(df_macro.index.max(), df_crypto.index.max())
        full_index = pd.date_range(start, end, freq='D')
        df_macro = df_macro.reindex(full_index).ffill()
        df_macro.index.name = 'datetime'

    # Onchain (optional)
    onchain_path = paths.get('onchain_csv')
    df_onchain = pd.DataFrame()
    if onchain_path and os.path.exists(onchain_path):
        df_onchain = pd.read_csv(onchain_path, parse_dates=['date'], index_col='date')
        df_onchain.index = pd.to_datetime(df_onchain.index, errors='coerce')
        df_onchain = df_onchain[~df_onchain.index.isna()]
        if df_onchain.index.has_duplicates:
            df_onchain = df_onchain.groupby(df_onchain.index).last()
        start = min(df_onchain.index.min(), df_crypto.index.min())
        end = max(df_onchain.index.max(), df_crypto.index.max())
        full_index = pd.date_range(start, end, freq='D')
        df_onchain = df_onchain.reindex(full_index).ffill()
        df_onchain.index.name = 'datetime'

    # Features from crypto
    df_features = compute_technical_indicators(df_crypto)
    if df_features.index.name is None:
        df_features.index.name = 'datetime'

    # Merge
    df_all = merge_macro_onchain(df_features, df_macro, df_onchain)
    df_all.sort_index(inplace=True)

    # Keep rows that contain crypto-derived features (allow macro/onchain NaN)
    required_cols = [c for c in df_all.columns if c.upper().startswith('BTC') or c.upper().startswith('ETH')]
    if required_cols:
        df_all = df_all.dropna(axis=0, subset=required_cols, how='any')
    else:
        warnings.warn("No crypto feature columns detected after compute_technical_indicators().")

    # Save processed CSV (write header even if empty)
    processed_dir = paths.get('data_processed')
    if processed_dir is None:
        raise KeyError("config['paths']['data_processed'] is required")
    os.makedirs(processed_dir, exist_ok=True)
    processed_path = os.path.join(processed_dir, "crypto_processed.csv")

    if df_all.empty:
        cols = []
        if not df_features.empty:
            cols.extend(df_features.columns.tolist())
        if not df_macro.empty:
            cols.extend([c for c in df_macro.columns if c not in cols])
        if not df_onchain.empty:
            cols.extend([c for c in df_onchain.columns if c not in cols])
        if not cols:
            cols = ["BTCUSDT_open", "BTCUSDT_high", "BTCUSDT_low", "BTCUSDT_close", "BTCUSDT_volume", "BTC_logp", "BTC_ret"]
        df_empty = pd.DataFrame(columns=cols)
        df_empty.index.name = 'datetime'
        df_empty.to_csv(processed_path)
        print(f"⚠️ preprocess_data: result is empty — wrote header to {processed_path}")
    else:
        df_all.to_csv(processed_path)
        print(f"Processed data saved to {processed_path} with {len(df_all)} rows")

    return df_all
