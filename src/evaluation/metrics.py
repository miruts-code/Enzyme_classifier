"""Evaluation metrics: aggregate + per-class + bootstrap CIs."""
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, f1_score, matthews_corrcoef,
    precision_recall_curve, auc,
    precision_score, recall_score,
    classification_report,
)
from config import N_BOOTSTRAP, SEED


def macro_f1(y_true, y_pred):
    return f1_score(y_true, y_pred, average="macro", zero_division=0)


def weighted_f1(y_true, y_pred):
    return f1_score(y_true, y_pred, average="weighted", zero_division=0)


def macro_precision(y_true, y_pred):
    return precision_score(y_true, y_pred, average="macro", zero_division=0)


def macro_recall(y_true, y_pred):
    return recall_score(y_true, y_pred, average="macro", zero_division=0)


def mcc(y_true, y_pred):
    return matthews_corrcoef(y_true, y_pred)


def accuracy(y_true, y_pred):
    return accuracy_score(y_true, y_pred)


def macro_auprc(y_true, y_proba, classes):
    aucs = []
    for i, c in enumerate(classes):
        y_bin = (np.asarray(y_true) == c).astype(int)
        p = y_proba[:, i]
        prec, rec, _ = precision_recall_curve(y_bin, p)
        aucs.append(auc(rec, prec))
    return float(np.mean(aucs))


def full_report(y_true, y_pred, y_proba, classes):
    aggregate = {
        "accuracy":        accuracy(y_true, y_pred),
        "macro_precision": macro_precision(y_true, y_pred),
        "macro_recall":    macro_recall(y_true, y_pred),
        "macro_f1":        macro_f1(y_true, y_pred),
        "weighted_f1":     weighted_f1(y_true, y_pred),
        "mcc":             mcc(y_true, y_pred),
        "macro_auprc": (
            macro_auprc(y_true, y_proba, classes)
            if y_proba is not None else None
        ),
    }
    cr = classification_report(y_true, y_pred, zero_division=0, output_dict=True)
    per_class = pd.DataFrame(cr).T
    per_class.index.name = "class"
    aggregate["per_class"] = per_class
    return aggregate


def bootstrap_metric_ci(y_true, y_pred, metric_fn=macro_f1,
                        n_boot=N_BOOTSTRAP, random_state=SEED):
    """Bootstrap a 95% CI for a metric. Default: macro F1."""
    rng = np.random.RandomState(random_state)
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    n = len(y_true)
    scores = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.randint(0, n, n)
        scores[i] = metric_fn(y_true[idx], y_pred[idx])
    return {
        "point_estimate": float(metric_fn(y_true, y_pred)),
        "ci_low":  float(np.percentile(scores, 2.5)),
        "ci_high": float(np.percentile(scores, 97.5)),
    }
