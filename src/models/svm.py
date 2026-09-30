"""SVMs with tune-on-subsample / refit-on-full-train workflow."""
from __future__ import annotations

import numpy as np
from sklearn.svm import LinearSVC, SVC
from sklearn.model_selection import RandomizedSearchCV
from sklearn.calibration import CalibratedClassifierCV
from src.models._common import stratified_subsample
import config


SVC_PARAM_DIST = {"C": [0.1, 1.0, 10.0]}
RBF_SVC_PARAM_DIST = {"C": [0.1, 1.0, 10.0], "gamma": ["scale", 0.01]}

MAX_ITER_LINEAR_SVC = 20000



def tune_linear_svm(X_train, y_train, params=None):
    if params is not None:
        base = LinearSVC(
            random_state=config.SEED, max_iter=MAX_ITER_LINEAR_SVC,
            class_weight="balanced", dual=False,**params,
        )
        calibrated = CalibratedClassifierCV(base, cv=config.CV_FOLDS)
        calibrated.fit(X_train, y_train)
        return calibrated, None, params

    X_sub, y_sub = stratified_subsample(
        X_train, y_train, config.TUNING_SUBSAMPLE_SIZE
    )
    base = LinearSVC(
        random_state=config.SEED, max_iter=MAX_ITER_LINEAR_SVC,
        class_weight="balanced", dual=False
    )
    search = RandomizedSearchCV(
        base, SVC_PARAM_DIST,
        n_iter=config.N_ITER_RANDOM_SEARCH, cv=config.CV_FOLDS,
        scoring=config.SCORING, random_state=config.SEED,
        n_jobs=1, verbose=0, error_score="raise",
    )
    search.fit(X_sub, y_sub)
    best_params = search.best_params_
    best_score = float(search.best_score_)

    final = LinearSVC(
        random_state=config.SEED, max_iter=MAX_ITER_LINEAR_SVC,
        class_weight="balanced", dual=False,**best_params,
    )
    calibrated = CalibratedClassifierCV(final, cv=config.CV_FOLDS)
    calibrated.fit(X_train, y_train)
    return calibrated, best_score, best_params


def tune_rbf_svm(X_train, y_train, params=None):
    if params is not None:
        X_sub, y_sub = stratified_subsample(
            X_train, y_train, config.SVM_SUBSAMPLE
        )
        final = SVC(
            kernel="rbf", probability=config.RBF_SVM_PROBABILITY,
            random_state=config.SEED, class_weight="balanced", **params,
        )
        final.fit(X_sub, y_sub)
        return final, None, params

    X_sub, y_sub = stratified_subsample(
        X_train, y_train, config.SVM_SUBSAMPLE
    )
    base = SVC(
        kernel="rbf", probability=config.RBF_SVM_PROBABILITY,
        random_state=config.SEED, class_weight="balanced",
    )
    search = RandomizedSearchCV(
        base, RBF_SVC_PARAM_DIST,
        n_iter=config.N_ITER_RANDOM_SEARCH, cv=config.CV_FOLDS,
        scoring=config.SCORING, random_state=config.SEED,
        n_jobs=1, verbose=0, error_score="raise",
    )
    search.fit(X_sub, y_sub)
    best_params = search.best_params_
    best_score = float(search.best_score_)

    final = SVC(
        kernel="rbf", probability=config.RBF_SVM_PROBABILITY,
        random_state=config.SEED, class_weight="balanced", **best_params,
    )
    final.fit(X_sub, y_sub)
    return final, best_score, best_params
