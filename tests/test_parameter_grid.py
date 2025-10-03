import json
from datetime import datetime
import yaml
import os

from models.xgboost_model import train_xgb
from models.dnn_model import train_dnn

# Load base config
cfg = yaml.safe_load(open("config.yaml"))

# small parameter grid
xgb_grid = [
    {"max_depth": 2, "n_estimators": 50},
    {"max_depth": 3, "n_estimators": 100},
]
dnn_grid = [
    {"epochs": 3, "batch_size": 32},
    {"epochs": 5, "batch_size": 16},
]

results = []
for xgb_params in xgb_grid:
    for dnn_params in dnn_grid:
        cfg_local = dict(cfg)
        cfg_local.setdefault("training", {})
        cfg_local["training"]["xgb"] = xgb_params
        cfg_local["training"]["dnn"] = dnn_params
        # keep small validation size to speed up
        cfg_local["training"]["test_size"] = 0.2

        print(f"Running grid: XGB {xgb_params} DNN {dnn_params}")
        xgb_model = train_xgb(cfg_local)
        dnn_model = train_dnn(cfg_local)
        results.append({"xgb": xgb_params, "dnn": dnn_params, "timestamp": datetime.utcnow().isoformat()})

# Save grid results summary
os.makedirs("logs", exist_ok=True)
out = os.path.join("logs", f"param_grid_results_{datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')}.json")
with open(out, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)
print(f"Saved grid run summary to {out}")