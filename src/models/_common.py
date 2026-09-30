"""Shared helpers for the model tuning modules."""
from __future__ import annotations

import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit

import config


def stratified_subsample(X, y, size):
    """Return a stratified subsample of (X, y) with `size` rows.

    If the data already has `size` rows or fewer, it is returned unchanged.
    """
    if len(y) <= size:
        return X, np.asarray(y)
    splitter = StratifiedShuffleSplit(
        n_splits=1, train_size=size, random_state=config.SEED
    )
    idx, _ = next(splitter.split(X, y))
    X_sub = X[idx] if hasattr(X, "__getitem__") else np.asarray(X)[idx]
    return X_sub, np.asarray(y)[idx]