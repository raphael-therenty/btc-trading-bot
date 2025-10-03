import os
import pandas as pd
from binance.client import Client
from datetime import datetime

def download_binance_data(config):
    symbol = config['binance']['symbol']
    interval = config['binance']['interval']
    start = config['binance']['historical_start']
    api_key = config['binance']['api_key']
    api_secret = config['binance']['api_secret']
    
    client = Client(api_key, api_secret)
    klines = client.get_historical_klines(symbol, interval, start_str=start, end_str=datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"))
    
    columns = [
        'open_time', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_volume', 'num_trades',
        'taker_buy_base_volume', 'taker_buy_quote_volume', 'ignore'
    ]
    df = pd.DataFrame(klines, columns=columns)
    df['datetime'] = pd.to_datetime(df['open_time'], unit='ms')
    df.set_index('datetime', inplace=True)
    df = df[['open','high','low','close','volume']].astype(float)
    df.rename(columns={
        'open': f'{symbol}_open',
        'high': f'{symbol}_high',
        'low': f'{symbol}_low',
        'close': f'{symbol}_close',
        'volume': f'{symbol}_volume'
    }, inplace=True)
    
    path = config['paths']['crypto_csv']
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path)
    print(f"Saved Binance data to {path} ({len(df)} rows)")
    return df
