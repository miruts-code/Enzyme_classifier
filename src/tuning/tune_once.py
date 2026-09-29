"""
Tune each model ONCE and save the winning hyperparameters to
cache/tuned_params.json.

Everything is read from config.py:
    MODE, TUNING_SUBSAMPLE_SIZE, N_ITER_RANDOM_SEARCH, CV_FOLDS,
    SVM_SUBSAMPLE, SCORING, SEED

Run this once. The main pipeline then loads the JSON and skips tuning.
"""
import json
import time
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import RandomizedSearchCV, StratifiedShuffleSplit
from sklearn.svm import LinearSVC

import config
from config import (
    SEED, CV_FOLDS, SCORING, CACHE,
    TUNING_SUBSAMPLE_SIZE, N_ITER_RANDOM_SEARCH, SVM_SUBSAMPLE,
)
from src.models.rf import tune_rf, RF_PARAM_DIST
from src.models.lgbm import tune_lgbm, LGBM_PARAM_DIST
from src.models.svm import tune_rbf_svm, SVC_PARAM_DIST  


TUNED_PARAMS_PATH = CACHE / "tuned_params.json"


# ------------------------------------------------------------------ #
# Helper                                                              #
# ------------------------------------------------------------------ #
def _sample(df, n):
    """Stratified sample of n rows from a DataFrame with a 'primary' column."""
    if len(df) <= n:
        return df
    splitter = StratifiedShuffleSplit(
        n_splits=1, train_size=n, random_state=SEED,
    )
    idx, _ = next(splitter.split(df, df["primary"]))
    return df.iloc[idx].reset_index(drop=True)


def _run(name, tune_fn, X, y):
    """Run an imported tuner, print timing/score and return best params."""
    print(f"  [{name}] tuning on {len(y)} rows "
          f"(n_iter={N_ITER_RANDOM_SEARCH}, cv={CV_FOLDS})...")
    t0 = time.time()
    _, score, best_params = tune_fn(X, y)     # (model, cv_score, best_params)
    print(f"  [{name}] done in {time.time()-t0:.1f}s — cv={score:.4f}")
    return best_params



def tune_linear_svm_once(X, y):
    n_iter = min(N_ITER_RANDOM_SEARCH, len(SVC_PARAM_DIST["C"]))
    print(f"  [LinearSVM] tuning on {len(y)} rows "
          f"(n_iter={n_iter}, cv={CV_FOLDS})...")
    t0 = time.time()
    search = RandomizedSearchCV(
        LinearSVC(random_state=SEED, max_iter=5000, class_weight="balanced"),
        param_distributions=SVC_PARAM_DIST,
        n_iter=n_iter, cv=CV_FOLDS, scoring=SCORING,
        random_state=SEED, n_jobs=1,
    )
    search.fit(X, y)
    print(f"  [LinearSVM] done in {time.time()-t0:.1f}s — cv={search.best_score_:.4f}")
    return search.best_params_


# ------------------------------------------------------------------ #
# Main                                                                #
# ------------------------------------------------------------------ #
def main():
 
    print("=" * 60)
    print(f"MODE: {getattr(config, 'MODE', 'unknown')}")
    print(f"  TUNING_SUBSAMPLE_SIZE  = {TUNING_SUBSAMPLE_SIZE}")
    print(f"  N_ITER_RANDOM_SEARCH   = {N_ITER_RANDOM_SEARCH}")
    print(f"  CV_FOLDS               = {CV_FOLDS}")
    print(f"  SVM_SUBSAMPLE          = {SVM_SUBSAMPLE}")
    print(f"  SEED                   = {SEED}")
    print(f"  SCORING                = {SCORING}")
    print("=" * 60)

    print("\nLoading data...")
    train_raw, _, _ = load_raw()
    train_df = prepare(clean(train_raw, verbose=False))

    print(f"Drawing {TUNING_SUBSAMPLE_SIZE}-row stratified sample...")
    sample = _sample(train_df, TUNING_SUBSAMPLE_SIZE)
    print(f"  sample size: {len(sample)}")
    print("  class distribution:")
    print(sample["primary"].value_counts().sort_index().to_string())

    print("\nBuilding AAC features...")
    X = build_aac(sample["seq"].tolist())
    y = sample["primary"].values
    print(f"  X shape: {X.shape}")

    print()
    params = {}
    params["RF"]        = _run("RF", tune_rf, X, y)
    params["LGBM"]      = _run("LGBM", tune_lgbm, X, y)
    params["LinearSVM"] = tune_linear_svm_once(X, y)
    params["RBF_SVM"]   = _run("RBF-SVM", tune_rbf_svm, X, y)

    with open(TUNED_PARAMS_PATH, "w") as f:
        json.dump(params, f, indent=2)

    print(f"\nSaved tuned params to {TUNED_PARAMS_PATH}")
    print(json.dumps(params, indent=2))


if __name__ == "__main__":
    main()