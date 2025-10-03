import os
import pandas as pd
from utils.config_loader import load_config
from data_scripts.download_binance import download_binance_data
from data_scripts.download_macro import download_macro_data
from data_scripts.preprocess_data import preprocess_data
from models.xgboost_model import train_xgb, predict_xgb
from models.dnn_model import train_dnn, predict_dnn
from models.garch_model import forecast_garch
from utils.backtest import backtest_strategy

def run_pipeline():
    cfg = load_config("config.yaml")
    
    os.makedirs(cfg['paths']['data_raw'], exist_ok=True)
    
    # Download data
    download_binance_data(cfg)
    download_macro_data(cfg)
    
    # Preprocess
    df_all = preprocess_data(cfg)
    
    # Train models
    xgb_model = train_xgb(cfg)
    dnn_model = train_dnn(cfg)
    
    # Predict
    xgb_prob = predict_xgb(cfg, xgb_model)
    dnn_prob = predict_dnn(cfg, dnn_model)
    
    # GARCH volatility
    garch_vol = forecast_garch(cfg)
    
    # Save outputs
    os.makedirs(cfg['paths']['outputs_models'], exist_ok=True)
    xgb_prob.to_csv(cfg['paths']['btc_xgb_prob'])
    dnn_prob.to_csv(cfg['paths']['btc_dnn_prob'])
    garch_vol.to_csv(cfg['paths']['btc_garch_forecast'])
    
    # Combine signals and backtest
    signals = pd.DataFrame({
        'xgb_prob': xgb_prob,
        'dnn_prob': dnn_prob,
        'garch_vol': garch_vol
    }).dropna()
    signals.to_csv(cfg['paths']['signals_csv'])
    
    backtest_strategy(cfg, signals)
    print("Pipeline completed successfully.")

if __name__ == "__main__":
    run_pipeline()
