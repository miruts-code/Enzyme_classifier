"""Plots: confusion matrix, ROC, feature importance."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize
from config import OUT
from src.data.labels import label_to_name


def plot_confusion(y_true, y_pred, classes, name):
    cm = confusion_matrix(y_true, y_pred, labels=classes, normalize="true")
    tick_labels = [f"{c} - {label_to_name(c)}" for c in classes]

    fig, ax = plt.subplots(figsize=(9, 7))
    im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)

    ax.set_xticks(range(len(classes)))
    ax.set_xticklabels(tick_labels, rotation=30, ha="right", fontsize=9)
    ax.set_yticks(range(len(classes)))
    ax.set_yticklabels(tick_labels, fontsize=9)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("Actual class")
    ax.set_title(f"Confusion matrix (row-normalized) - {name}")

    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, f"{cm[i, j]:.2f}", ha="center", va="center",
                    color="white" if cm[i, j] > 0.5 else "black", fontsize=9)

    fig.colorbar(im, fraction=0.046, pad=0.04, label="Fraction of true class")
    fig.tight_layout()
    fig.savefig(OUT / f"confusion_{name}.png", dpi=130)
    plt.close(fig)


def plot_roc(y_true, y_proba, classes, name):
    y_bin = label_binarize(y_true, classes=classes)
    fig, ax = plt.subplots(figsize=(9, 7))
    for i, c in enumerate(classes):
        if y_bin[:, i].sum() == 0:
            continue
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_proba[:, i])
        ax.plot(fpr, tpr, label=f"{label_to_name(c)} (AUC={auc(fpr, tpr):.3f})",
                linewidth=1.8)
    ax.plot([0, 1], [0, 1], "k--", linewidth=1)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title(f"One-vs-rest ROC - {name}")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / f"roc_{name}.png", dpi=130)
    plt.close(fig)


def plot_importance(model, feature_names, name, top=30):
    if not hasattr(model, "feature_importances_"):
        return
    imp = model.feature_importances_
    order = np.argsort(imp)[::-1][:top]
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.barh(range(len(order)), imp[order][::-1])
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([feature_names[i] for i in order][::-1], fontsize=8)
    ax.set_xlabel("Importance")
    ax.set_title(f"Feature importance - {name}")
    fig.tight_layout()
    fig.savefig(OUT / f"importance_{name}.png", dpi=130)
    plt.close(fig)
