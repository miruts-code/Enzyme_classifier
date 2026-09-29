"""
Holdout stacking: fit the meta-learner on the validation set.

Base models are trained on train only (never on val), so their
predictions on val are honest out-of-sample outputs. We fit a small
Logistic Regression meta-learner on those val predictions, then apply
it to the test predictions.

"""
import numpy as np
from sklearn.linear_model import LogisticRegression
from config import SEED


def holdout_stacking(val_probas, test_probas, y_val, seed=SEED):
    """Fit meta-learner on validation probabilities.

    Parameters
    ----------
    val_probas : list of (n_val, n_classes) arrays
    test_probas : list of (n_test, n_classes) arrays
    y_val : array of validation labels

    Returns
    -------
    (pred, proba) on test.
    """
    # log-probabilities make LogisticRegression behave better on
    # probabilities pinned near 0 or 1
    lg = lambda p: np.log(np.clip(p.astype(np.float64), 1e-6, 1.0))
    Zva = np.hstack([lg(p) for p in val_probas])
    Zte = np.hstack([lg(p) for p in test_probas])

    meta = LogisticRegression(
        class_weight="balanced",
        max_iter=2000,
        random_state=seed,
    ).fit(Zva, y_val)

    return meta.predict(Zte), meta.predict_proba(Zte)
