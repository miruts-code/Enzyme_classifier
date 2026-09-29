"""Random Forest with tune-on-subsample / refit-on-full-train workflow."""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, StratifiedShuffleSplit

import config


RF_PARAM_DIST = {
    "n_estimators":     [100, 200, 300],
    "max_depth":        [10, 20],
    "min_samples_leaf": [1, 2, 4],
    "max_features":     ["sqrt", "log2"],
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


def tune_rf(X_train, y_train, params=None):
    """Return (model, best_cv_score, best_params)."""
    if params is not None:
        # Pipeline path: no search, use all cores.
        final = RandomForestClassifier(
            random_state=config.SEED,
            n_jobs=-1,
            class_weight="balanced",
            **params,
        )
        final.fit(X_train, y_train)
        return final, None, params

    # Tuning path: n_jobs=1 to avoid nested parallelism with the search.
    X_sub, y_sub = _stratified_subsample(
        X_train, y_train, config.TUNING_SUBSAMPLE_SIZE
    )
    base = RandomForestClassifier(
        random_state=config.SEED, n_jobs=1, class_weight="balanced",
    )
    search = RandomizedSearchCV(
        base,
        param_distributions=RF_PARAM_DIST,
        n_iter=config.N_ITER_RANDOM_SEARCH,
        cv=config.CV_FOLDS,
        scoring=config.SCORING,
        random_state=config.SEED,
        n_jobs=1,
        verbose=0,
        error_score="raise",
    )
    search.fit(X_sub, y_sub)
    best_params = search.best_params_
    best_score = float(search.best_score_)

    final = RandomForestClassifier(
        random_state=config.SEED,
        n_jobs=-1,
        class_weight="balanced",
        **best_params,
    )
    final.fit(X_train, y_train)
    return final, best_score, best_params
