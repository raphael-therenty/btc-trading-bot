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

def _determine_windows(cfg, proc_df):
    train_cfg = cfg.get('training', {})
    # explicit window
    if train_cfg.get('train_start') or train_cfg.get('train_end'):
        start = pd.to_datetime(train_cfg.get('train_start')) if train_cfg.get('train_start') else proc_df.index[0]
        end = pd.to_datetime(train_cfg.get('train_end')) if train_cfg.get('train_end') else proc_df.index[-1]
        # holdout is everything after end
        holdout_start = (end + pd.Timedelta(days=1)) if end < proc_df.index[-1] else None
        holdout_end = proc_df.index[-1] if holdout_start is not None else None
        return start, end, holdout_start, holdout_end
    # fallback to ratio split
    test_size = float(train_cfg.get('test_size', 0.2))
    n = len(proc_df)
    split = max(1, int(n * (1 - test_size)))
    train_start = proc_df.index[0]
    train_end = proc_df.index[split - 1]
    holdout_start = proc_df.index[split]
    holdout_end = proc_df.index[-1]
    return train_start, train_end, holdout_start, holdout_end

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
    
    # Predict on full processed data
    xgb_prob = predict_xgb(cfg, xgb_model)
    dnn_prob = predict_dnn(cfg, dnn_model)
    garch_vol = forecast_garch(cfg)
    
    # Save outputs
    os.makedirs(cfg['paths']['outputs_models'], exist_ok=True)
    xgb_prob.to_csv(cfg['paths']['btc_xgb_prob'])
    dnn_prob.to_csv(cfg['paths']['btc_dnn_prob'])
    garch_vol.to_csv(cfg['paths']['btc_garch_forecast'])
    
    # Combine signals (do not drop rows based on garch by default)
    signals = pd.DataFrame({
        'xgb_prob': xgb_prob,
        'dnn_prob': dnn_prob,
        'garch_vol': garch_vol
    })
    signals.to_csv(cfg['paths']['signals_csv'])
    
    # determine training and holdout windows and filter signals to holdout for backtest
    proc_df = pd.read_csv(os.path.join(cfg['paths']['data_processed'], "crypto_processed.csv"),
                           parse_dates=['datetime'], index_col='datetime')
    train_start, train_end, holdout_start, holdout_end = _determine_windows(cfg, proc_df)
    if holdout_start is None:
        print("No holdout period available (training window reaches dataset end). Backtest will not run.")
        return

    signals_holdout = signals.loc[(signals.index >= holdout_start) & (signals.index <= holdout_end)].dropna(how='all')
    if signals_holdout.empty:
        print("No signals in holdout period — nothing to backtest.")
        return

    # Run backtest on holdout only
    trades_df, metrics = backtest_strategy(cfg, signals_holdout, proc_df=proc_df, backtest_window=(holdout_start, holdout_end))
    print("Backtest metrics summary:")
    print(f"trades={metrics['n_trades']} realized={metrics['n_realized']} wins={metrics['wins']} win_rate={metrics['win_rate']}")
    print(f"cum_pnl=${metrics['cum_pnl']:.2f}  cum_return={metrics['cum_return']:.4f} sharpe={metrics['strategy_sharpe']}")
    print("Pipeline completed successfully.")

if __name__ == "__main__":
    run_pipeline()