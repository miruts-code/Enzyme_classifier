"""Load tuned hyperparameters from cache/tuned_params.json."""
import json
from config import CACHE

TUNED_PARAMS_PATH = CACHE / "tuned_params.json"


def load_tuned_params():
    if not TUNED_PARAMS_PATH.exists():
        print(f"[params] {TUNED_PARAMS_PATH} not found — will tune from scratch.")
        return {}
    with open(TUNED_PARAMS_PATH) as f:
        return json.load(f)


def get_params_for(model_name, fallback=None):
    all_params = load_tuned_params()
    return all_params.get(model_name, fallback)
