import os
import yaml

def load_config(path=None):
    """Load configuration from config.yaml."""
    if path is None:
        here = os.path.dirname(__file__)
        path = os.path.join(here, "..", "config.yaml")
    path = os.path.abspath(path)
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg
