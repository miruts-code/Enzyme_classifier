"""
SHAP explanations for tree-based models (LightGBM).
"""
from __future__ import annotations

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _to_class_list(shap_values, n_features):
    """Normalize any shap output format to a list of (n_samples, n_features)."""
    if isinstance(shap_values, list):
        return [np.asarray(s) for s in shap_values]

    arr = np.asarray(shap_values)
    if arr.ndim == 2:
        return [arr]
    if arr.ndim == 3:
        if arr.shape[1] == n_features:
            return [arr[:, :, c] for c in range(arr.shape[2])]
        if arr.shape[2] == n_features:
            return [arr[c] for c in range(arr.shape[0])]
    raise ValueError(f"Unexpected SHAP shape {arr.shape} for {n_features} features")


def mean_abs_importance(shap_values, n_features):
    """Mean |SHAP| over samples, averaged over classes."""
    per_class = _to_class_list(shap_values, n_features)
    return np.mean([np.abs(s).mean(axis=0) for s in per_class], axis=0)


def explain_tree_model(model, X_sample):
    import shap
    explainer = shap.TreeExplainer(model)
    return explainer, explainer.shap_values(X_sample)


def plot_shap_summary(shap_values, X_sample, feature_names=None,
                      max_display=20, save_path=None):
    n_features = X_sample.shape[1]
    if feature_names is None:
        feature_names = [f"f{i}" for i in range(n_features)]

    imp = mean_abs_importance(shap_values, n_features)
    top = np.argsort(imp)[::-1][:max_display][::-1]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh([feature_names[i] for i in top], imp[top], color="#4C72B0")
    ax.set_xlabel("mean |SHAP| (averaged over classes)")
    ax.set_title("LightGBM SHAP feature importance")
    ax.grid(True, alpha=0.3, axis="x")
    fig.tight_layout()

    try:
        if save_path is not None:
            tmp = save_path.with_name(save_path.stem + ".tmp.png")
            fig.savefig(tmp, dpi=150, bbox_inches="tight")
            assert tmp.stat().st_size > 5_000, "SHAP figure looks empty"
            os.replace(tmp, save_path)
    finally:
        plt.close(fig)


def compare_with_builtin_importance(model, shap_values, feature_names, top_n=20):
    if not hasattr(model, "feature_importances_"):
        return {"error": "model has no feature_importances_"}

    shap_imp = mean_abs_importance(shap_values, len(feature_names))
    builtin_imp = np.asarray(model.feature_importances_)

    shap_top = [feature_names[i] for i in np.argsort(shap_imp)[::-1][:top_n]]
    builtin_top = [feature_names[i] for i in np.argsort(builtin_imp)[::-1][:top_n]]

    return {
        "shap_top_features": shap_top,
        "builtin_top_features": builtin_top,
        "overlap_at_top_n": len(set(shap_top) & set(builtin_top)),
        "top_n": top_n,
    }
