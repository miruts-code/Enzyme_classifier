"""
Voting ensembles.

- make_hard_voting / make_soft_voting: sklearn VotingClassifier wrappers.
  NOTE: sklearn's VotingClassifier clones and REFITS its estimators when
  .fit() is called — passing fitted models to it does not save time.

- combine_hard_vote / combine_soft_vote: array-based alternatives that
  combine ALREADY-COMPUTED predictions/probabilities. No refitting at all.
"""
from __future__ import annotations

import numpy as np
from scipy import stats
from sklearn.ensemble import VotingClassifier


# --- sklearn wrappers (kept for compatibility) ---
def make_hard_voting(estimators):
    return VotingClassifier(estimators=estimators, voting="hard", n_jobs=-1)


def make_soft_voting(estimators):
    return VotingClassifier(estimators=estimators, voting="soft", n_jobs=-1)


# --- Array-based combiners — no refitting ---
def combine_hard_vote(pred_list):
    """Majority vote over already-computed predictions."""
    stacked = np.column_stack(pred_list)
    return stats.mode(stacked, axis=1, keepdims=False).mode


def combine_soft_vote(proba_list, classes):
    """Average probabilities across already-computed probability arrays.

    Class column order must match across all arrays.
    """
    avg = np.mean(proba_list, axis=0)
    classes_arr = np.asarray(classes)
    pred = classes_arr[np.argmax(avg, axis=1)]
    return pred, avg
