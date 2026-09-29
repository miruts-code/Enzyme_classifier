"""LightGBM with tune-on-subsample / refit-on-full-train workflow."""
from __future__ import annotations

import numpy as np
from sklearn.model_selection import RandomizedSearchCV, StratifiedShuffleSplit
from sklearn.utils.class_weight import compute_sample_weight
from lightgbm import LGBMClassifier

import config


LGBM_PARAM_DIST = {
    "n_estimators":  [100, 200, 300],
    "num_leaves":    [10, 31, 63],
    "learning_rate": [0.05, 0.1, 0.15],
}


def _stratified_subsample(X, y, size):
    if len(y) <= size:
        return X, np.asarray(y)
    splitter = StratifiedShuffleSplit(
        n_splits=1, train_size=size, random_state=config.SEED
    )
    idx, _ = next(splitter.split(X, y))
    X_sub = X[idx] if hasattr(X, "__getitem__") else np.asarray(X)[idx]
    return X_sub, np.asarray(y)[idx]


def tune_lgbm(X_train, y_train, params=None):
    """Return (model, best_cv_score, best_params)."""
    if params is not None:
        sw_full = compute_sample_weight("balanced", y_train)
        final = LGBMClassifier(
            random_state=config.SEED,
            n_jobs=1,                # single-threaded to avoid thrashing
            verbose=-1,
            **params,
        )
        final.fit(X_train, y_train, sample_weight=sw_full)
        return final, None, params

    X_sub, y_sub = _stratified_subsample(
        X_train, y_train, config.TUNING_SUBSAMPLE_SIZE
    )
    sw_sub = compute_sample_weight("balanced", y_sub)
    sw_full = compute_sample_weight("balanced", y_train)

    base = LGBMClassifier(random_state=config.SEED, n_jobs=1, verbose=-1)
    search = RandomizedSearchCV(
        base,
        param_distributions=LGBM_PARAM_DIST,
        n_iter=config.N_ITER_RANDOM_SEARCH,
        cv=config.CV_FOLDS,
        scoring=config.SCORING,
        random_state=config.SEED,
        n_jobs=1,
        verbose=0,
        error_score="raise",
    )
    search.fit(X_sub, y_sub, sample_weight=sw_sub)
    best_params = search.best_params_
    best_score = float(search.best_score_)

    final = LGBMClassifier(
        random_state=config.SEED,
        n_jobs=1,
        verbose=-1,
        **best_params,
    )
    final.fit(X_train, y_train, sample_weight=sw_full)
    return final, best_score, best_params
