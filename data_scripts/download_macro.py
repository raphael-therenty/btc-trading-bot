import os
import warnings
from datetime import datetime
import pandas as pd
import requests

try:
    from fredapi import Fred
except Exception:
    Fred = None


def download_macro_data(config):
    """
    Download or load macro data. Ensures start date 1954-07-01, robustly parses BEA TimePeriod,
    removes/aggregates duplicate dates and reindexes daily with forward-fill.
    Returns a DataFrame indexed by Date (daily).
    """
    paths = config.get('paths', {})
    output_path = paths.get('macro_csv')
    start_date = datetime(1954, 7, 1)

    if output_path is None:
        raise KeyError("config['paths']['macro_csv'] is required")

    # If file exists, load and sanitize
    if os.path.exists(output_path):
        df = pd.read_csv(output_path, parse_dates=["Date"], index_col="Date")
        df.index = pd.to_datetime(df.index, errors="coerce")
        df = df[~df.index.isna()]
        if df.index.has_duplicates:
            df = df.groupby(df.index).last()
        full_index = pd.date_range(start=start_date, end=pd.Timestamp.today(), freq="D")
        df = df.reindex(full_index).ffill()
        df.index.name = "Date"
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path)
        return df

    # Otherwise try to download
    print("Downloading macro data...")
    fred_key = config.get('_keys', {}).get('fred') or config.get('api_keys', {}).get('fred')
    fred = None
    try:
        if Fred is not None:
            fred = Fred(api_key=fred_key) if fred_key else Fred()
    except Exception:
        fred = None

    series_map = {
        "CPI_US": "CPIAUCSL",
        "Unemployment_US": "UNRATE",
        "FedFundsRate_US": "FEDFUNDS",
    }

    dfs = {}
    for name, code in series_map.items():
        try:
            if fred is not None:
                s = fred.get_series(code, observation_start=start_date)
                df_s = pd.DataFrame(s, columns=[name])
                df_s.index.name = "Date"
            else:
                url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={code}&observation_start={start_date.date()}"
                df_s = pd.read_csv(url, parse_dates=["DATE"], index_col="DATE")
                df_s.columns = [name]
                df_s.index.name = "Date"
            df_s.index = pd.to_datetime(df_s.index, errors="coerce")
            df_s = df_s[~df_s.index.isna()]
            if df_s.index.has_duplicates:
                df_s = df_s.groupby(df_s.index).last()
            dfs[name] = df_s
        except Exception as e:
            warnings.warn(f"Could not download {name} ({code}): {e}")
            dfs[name] = pd.DataFrame()

    # Download GDP from BEA
    bea_key = config.get('api_keys', {}).get('bea') or config.get('_keys', {}).get('bea')
    gdp_df = pd.DataFrame()
    try:
        if bea_key:
            url = (
                f"https://apps.bea.gov/api/data/?UserID={bea_key}"
                "&method=GetData&datasetname=NIPA&TableName=T10105"
                "&Frequency=Q&Year=ALL&ResultFormat=JSON"
            )
            resp = requests.get(url, timeout=30)
            resp.raise_for_status()
            payload = resp.json().get("BEAAPI", {}).get("Results", {}).get("Data", [])
            gdp_us = pd.DataFrame(payload)
            if not gdp_us.empty and "TimePeriod" in gdp_us.columns:
                tp = gdp_us["TimePeriod"].astype(str)

                def parse_timeperiod(x):
                    x = x.strip()
                    try:
                        if "Q" in x.upper():
                            norm = x.replace("-", "").upper()
                            return pd.Period(norm, freq="Q").to_timestamp()
                        return pd.to_datetime(x, errors="coerce")
                    except Exception:
                        return pd.NaT

                gdp_us["Date"] = tp.map(parse_timeperiod)
                gdp_us = gdp_us.dropna(subset=["Date"])
                if "DataValue" in gdp_us.columns:
                    gdp_us = gdp_us[["Date", "DataValue"]].rename(columns={"DataValue": "GDP_US"})
                elif "Value" in gdp_us.columns:
                    gdp_us = gdp_us[["Date", "Value"]].rename(columns={"Value": "GDP_US"})
                gdp_us = gdp_us.set_index("Date").sort_index()
                if gdp_us.index.has_duplicates:
                    gdp_us = gdp_us.groupby(gdp_us.index).last()
                gdp_df = gdp_us[gdp_us.index >= pd.Timestamp(start_date)]
    except Exception as e:
        warnings.warn(f"Failed to download BEA GDP: {e}")

    # Build combined df_us
    df_us = None
    for key, df_piece in dfs.items():
        if df_piece is None or df_piece.empty:
            continue
        if df_us is None:
            df_us = df_piece.copy()
        else:
            df_us = df_us.join(df_piece, how="outer")

    if not gdp_df.empty:
        if df_us is None:
            df_us = gdp_df.copy()
        else:
            df_us = df_us.join(gdp_df, how="outer")

    if df_us is None:
        df_us = pd.DataFrame()

    if not df_us.empty:
        df_us.index = pd.to_datetime(df_us.index, errors="coerce")
        df_us = df_us[~df_us.index.isna()]
        if df_us.index.has_duplicates:
            df_us = df_us.groupby(df_us.index).last()
        full_index = pd.date_range(start=start_date, end=pd.Timestamp.today(), freq="D")
        df_us = df_us.reindex(full_index).ffill()
        df_us.index.name = "Date"

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_us.to_csv(output_path)
    print(f"Macro data saved to {output_path} ({len(df_us)} rows)")
    return df_us
