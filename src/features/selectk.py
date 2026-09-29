"""
Feature selection for the high-dimensional tripeptide representation.

The exact k is not picked arbitrarily: sweep_k() fits a SelectKBest for
each candidate k in config.TRIPEPTIDE_K_GRID on train, trains a cheap
quick model, and scores macro F1 on the validation set. The winning k
is reported with its curve.
"""
from __future__ import annotations

import numpy as np
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.metrics import f1_score

import config


def fit_selector(X_train, y_train, k=None):
    """Fit SelectKBest(mutual_info_classif, k) on TRAINING data only."""
    k = k or config.SELECT_K
    selector = SelectKBest(mutual_info_classif, k=k)
    selector.fit(X_train, y_train)
    return selector


def sweep_k(X_train, y_train, X_val, y_val, quick_model_factory, k_grid=None):
    """Try each k in k_grid, score macro F1 on validation, return best k."""
    k_grid = k_grid or config.TRIPEPTIDE_K_GRID
    results = []
    for k in k_grid:
        selector = fit_selector(X_train, y_train, k)
        X_tr = selector.transform(X_train)
        X_va = selector.transform(X_val)
        model = quick_model_factory()
        model.fit(X_tr, y_train)
        preds = model.predict(X_va)
        score = float(f1_score(y_val, preds, average="macro", zero_division=0))
        results.append((k, score))
        print(f"  [sweep] k={k:>5}: val_macro_f1={score:.4f}")
    best_k = max(results, key=lambda r: r[1])[0]
    print(f"  [sweep] best k={best_k}")
    return best_k, results
