# Enzyme Function Classification from Protein Sequences

A complete supervised machine learning pipeline for predicting the top-level Enzyme Commission (EC) class of a protein from its amino-acid sequence. Trained and evaluated on the SwissProt-EC dataset.

**Current version:** 1.0.0
**Status:** Complete — full pipeline runs end-to-end, all outputs verified
**Dataset:** [DanielHesslow/SwissProt-EC](https://huggingface.co/datasets/DanielHesslow/SwissProt-EC) (~261K protein records)

---

## Table of Contents

1. [Task](#1-task)
2. [Dataset](#2-dataset)
3. [Project Structure](#3-project-structure)
4. [Requirements](#4-requirements)
5. [How to Run](#5-how-to-run)
6. [Pipeline Architecture](#6-pipeline-architecture)
7. [Feature Representations](#7-feature-representations)
8. [Models](#8-models)
9. [Ensembles](#9-ensembles)
10. [Evaluation Metrics](#10-evaluation-metrics)
11. [Outputs](#11-outputs)
12. [Caching and Resumability](#12-caching-and-resumability)
13. [Interpretability](#13-interpretability)
14. [Results Summary](#14-results-summary)
15. [Limitations](#15-limitations)
16. [Reproducibility](#16-reproducibility)

---

## 1. Task

**Given:** a protein amino-acid sequence, e.g.

```text
MKTAYIAKQRQISFVK...
```

**Predict:** the top-level EC class of the enzyme (1 through 7).

| **Class** | **Enzyme family** |
| :-------: | ----------------- |
|     1     | Oxidoreductases   |
|     2     | Transferases      |
|     3     | Hydrolases        |
|     4     | Lyases            |
|     5     | Isomerases        |
|     6     | Ligases           |
|     7     | Translocases      |

This is a **7-class supervised classification problem**.

---

## 2. Dataset

Source: **SwissProt-EC** on Hugging Face. SwissProt is a manually curated protein database; this dataset extracts the subset of proteins annotated with an EC number.

The dataset ships with predefined splits:

| **Split** | **Rows** |
| :-------- | -------: |
| train     |  208,823 |
| dev       |   26,724 |
| test      |   25,892 |

The pipeline uses these splits directly. The test set is used **once**, at the very end, for the final evaluation.

---

## 3. Project Structure

```text
This project structure is after a full pipeline run.
enzyme_classifier/
│
├── config.py
├── requirements.txt
├── README.md
│
├── cache/
│   └── tuned_params.json
│
├── outputs/
│   ├── metrics.csv
│   ├── per_class_all.csv
│   ├── bootstrap_cis.csv
│   ├── confusion_RF_AAC.png
│   ├── confusion_LGBM_AAC.png
│   ├── importance_RF_AAC.png
│   ├── importance_LGBM_AAC.png
│   ├── roc_LinearSVM_AAC.png
│   ├── shap_LGBM_AAC.png
│   │
│   ├── ckpt/
│   │   ├── metrics_AAC.csv
│   │   ├── metrics_DPC.csv
│   │   ├── metrics_TPC_k1000.csv
│   │   ├── metrics_SVD128.csv
│   │   ├── per_class_AAC.csv
│   │   ├── per_class_DPC.csv
│   │   ├── per_class_TPC_k1000.csv
│   │   ├── per_class_SVD128.csv
│   │   ├── bootstrap_AAC.csv
│   │   ├── bootstrap_DPC.csv
│   │   ├── bootstrap_TPC_k1000.csv
│   │   ├── bootstrap_SVD128.csv
│   │   ├── AAC.done
│   │   ├── DPC.done
│   │   ├── TPC_k1000.done
│   │   └── SVD128.done
│   │
│   └── proba/
│       ├── AAC_RF_*.npz
│       ├── AAC_LGBM_*.npz
│       ├── AAC_LinearSVM_*.npz
│       ├── AAC_RBF_SVM_*.npz
│       ├── DPC_RF_*.npz
│       └── ... 
│
├── notebooks/
│   └── run_pipeline.ipynb
│
└── src/
    ├── data/
    │   ├── loader.py
    │   └── labels.py
    │
    ├── features/
    │   ├── aac.py
    │   ├── dpc.py
    │   ├── tpc.py
    │   ├── selectk.py
    │   └── svd_embed.py
    │
    ├── models/
    │   ├── rf.py
    │   ├── lgbm.py
    │   └── svm.py
    │
    ├── ensembles/
    │   ├── voting.py
    │   └── stacking.py
    │
    ├── evaluation/
    │   ├── metrics.py
    │   ├── plots.py
    │   ├── error_analysis.py
    │   └── checks.py
    │
    ├── tuning/
    │   ├── tune_once.py
    │   └── params.py
    │
    ├── explainability/
    │   └── shap_analysis.py
    │
    └── pipeline.py
```

---

## 4. Requirements

Python 3.9+ with:

```text
datasets>=2.14
pandas>=2.0
numpy>=1.24
scikit-learn>=1.2
lightgbm>=4.0
matplotlib>=3.7
scipy>=1.10
shap>=0.44
```

### Install

```bash
pip install -r requirements.txt
```

On Google Colab or Kaggle, all except datasets, lightgbm, and shap come preinstalled. Install those three:

```bash
pip install -q datasets lightgbm shap
```

---

## 5. How to Run

The project can be run in two ways: from the command line, or from the example Colab notebook.

### Option A — From the command line

The pipeline has two phases.

### Phase 1 — One-time hyperparameter tuning

Runs once. Produces `cache/tuned_params.json`. Takes 5–15 minutes with the default 5,000-row tuning sample.

```bash
python -m src.tuning.tune_once
```

Or from a Python session:

```python
from src.tuning.tune_once import main

main()
```

#### What it does

* Samples `TUNING_SUBSAMPLE_SIZE` rows (default: 5,000) from the training split.
* Runs RandomizedSearchCV for RF, LGBM, LinearSVM, and RBF-SVM on that sample.
* Saves the winning hyperparameters to `cache/tuned_params.json`.

#### Why it runs once

The winning parameters are frozen. The pipeline refits each model on the full 208,823-row training set using those parameters.

### Phase 2 — The full pipeline

Trains, evaluates, and reports.

```bash
python -m src.pipeline
```

Or:

```python
from src.pipeline import run

result = run()
```

Runtime: ~1.5–2 hours on Colab CPU, depending on the machine.

Prerequisite: `cache/tuned_params.json` must exist (Phase 1 must have run).

### Resuming after a crash

The pipeline checkpoints after every feature set. If it crashes mid-run, rerun the same command. Completed feature sets are skipped automatically.

```python
from src.pipeline import run

run()   # resumes from where it stopped
```

---

### Option B — From the example notebook

`notebooks/run_pipeline.ipynb` is the working Colab notebook used to develop and run this project. It is provided as an example of how the pipeline is executed in practice on Google Colab.

The notebook walks through:

1. Mounting Google Drive.
2. Changing into the project directory.
3. Installing dependencies.
4. Running `tune_once` (Phase 1).
5. Running `pipeline.run()` (Phase 2).
6. Inspecting the outputs in `outputs/`.

To use it:

1. Upload `notebooks/run_pipeline.ipynb` to Google Colab.
2. Run the cells from top to bottom.
3. The first cell mounts Drive.
4. The second locates the project root.
5. The remaining cells run the two phases and display results.

---

## 6. Pipeline Architecture

```text
Load SwissProt-EC dataset (train / dev / test)
                ↓
Clean: drop invalid amino acids, missing labels, duplicates
                ↓
Verify: no cross-split leakage, class distribution preserved
                ↓
For each feature set (AAC → DPC → TPC_k1000 → SVD128):
    ├── Build feature matrix (cached to /content/cache/features)
    ├── Fit or load base models (RF, LGBM, LinearSVM, RBF-SVM)
    ├── Cache each model's val/test probabilities
    ├── Compute Hard Voting (combine predictions)
    ├── Compute Soft Voting (average probabilities)
    ├── Compute Holdout Stacking (fit meta-learner on val)
    ├── Save per-feature-set checkpoints (metrics, per-class, CIs)
    └── Free feature matrices, models, and intermediate arrays
                ↓
Assemble final metrics.csv, per_class_all.csv, bootstrap_cis.csv
```

Each feature set is built, evaluated, and freed in isolation. This bounds peak RAM to one feature set at a time.

---

## 7. Feature Representations

Four representations are used. Each captures a different level of sequence information.

### AAC — Amino Acid Composition (20 dims)

Frequency of each of the 20 standard amino acids.

Example:

```text
"MKTAY" → {M: 0.2, K: 0.2, T: 0.2, A: 0.2, Y: 0.2}
```

Captures global composition. Order-independent.

### DPC — Dipeptide Composition (400 dims)

Frequency of each adjacent amino-acid pair. 20 × 20 = 400 possible pairs.

Captures local order — which pairs appear together.

### TPC_k1000 — Tripeptide Composition with feature selection

Frequency of each 3-residue chunk. 20³ = 8,000 possible tripeptides.

The pipeline fits `SelectKBest(mutual_info_classif, k=1000)` on the training split to keep the 1,000 most informative tripeptides.

Captures short sequence motifs.

### SVD128 — TruncatedSVD embedding (128 dims)

The 8,000-dim TPC matrix compressed via `TruncatedSVD(n_components=128)`.

A dense, low-dim representation. Fitted on training data only.

---

## 8. Models

Each model is fitted once per feature set, using the hyperparameters from `cache/tuned_params.json`.

### Random Forest

Ensemble of decision trees. Robust to outliers. Handles non-linear patterns.

`class_weight="balanced"` for class imbalance.

### LightGBM

Gradient-boosted decision trees. Highest single-model accuracy in this pipeline.

Uses `sample_weight=compute_sample_weight("balanced", y)` for class balancing.

### LinearSVC (calibrated)

Linear support-vector classifier. `dual=False` (correct for tall data), `max_iter=20000`. Wrapped in `CalibratedClassifierCV` to produce `predict_proba`.

Fits on the full training set.

### RBF-SVM (standalone)

Exact kernel SVM on a stratified subsample (`SVM_SUBSAMPLE = 2,000` rows).

Not used in stacking (see below).

Provides a non-linear-kernel baseline for comparison.

---

## 9. Ensembles

Three ensembles combine the base models.

### Hard Voting

Majority vote over the predictions of RF, LGBM, and LinearSVM.

Computed via array combination — no retraining. Just counts votes.

### Soft Voting

Average of the probabilities from RF, LGBM, and RBF-SVM.

Also computed via array combination — no retraining.

### Stacking

A meta-learner (Logistic Regression) is trained on the validation set's base-model probabilities, then applied to test probabilities.

Design choice: the meta-learner is fit on validation predictions rather than via internal cross-validation. This eliminates the ~18 base-model refits that `sklearn.StackingClassifier(cv=5)` would perform, at the cost of using the validation set for meta-learner training.

Base models for stacking: RF, LGBM, LinearSVM (RBF-SVM excluded — its O(n²) refit dominates runtime without improving the ensemble).

---

## 10. Evaluation Metrics

Every model is evaluated on the test set with:

| Metric                        | Notes                                                  |
| ----------------------------- | ------------------------------------------------------ |
| Macro F1                      | Primary metric. Equal weight per class.                |
| MCC                           | Matthews Correlation Coefficient. Robust to imbalance. |
| Macro AUPRC                   | Area under precision-recall, macro-averaged.           |
| Accuracy                      | Reference only.                                        |
| Weighted F1                   | F1 weighted by class support.                          |
| Per-class precision/recall/F1 | Reported in `per_class_all.csv`.                       |

Additionally, bootstrap 95% confidence intervals are computed for the stacking ensemble's macro F1 (300 resamples).

---

## 11. Outputs

All outputs are saved under `outputs/`.

### Aggregate results

| File                | Content                                                                                                                                                                            |
| ------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `metrics.csv`       | Main results table — one row per model × feature set (28 rows). Columns: feature, model, accuracy, macro_precision, macro_recall, macro_f1, weighted_f1, mcc, macro_auprc, params. |
| `per_class_all.csv` | Per-class precision, recall, F1, support for every model.                                                                                                                          |
| `bootstrap_cis.csv` | 95% bootstrap CIs for the stacking ensemble's macro F1.                                                                                                                            |

### Plots (only for the target feature set — AAC)

The pipeline generates plots only for the feature set configured in `config.TARGET_FEATURE_FOR_PLOTS` (default: `"AAC"`). Set it to `None` to skip plots entirely.

| File                      | Content                                     |
| ------------------------- | ------------------------------------------- |
| `confusion_RF_AAC.png`    | Normalized confusion matrix for RF on AAC   |
| `confusion_LGBM_AAC.png`  | Normalized confusion matrix for LGBM on AAC |
| `importance_RF_AAC.png`   | Top-30 feature importance for RF            |
| `importance_LGBM_AAC.png` | Top-30 feature importance for LGBM          |
| `roc_LinearSVM_AAC.png`   | One-vs-rest ROC curves for LinearSVM        |
| `shap_LGBM_AAC.png`       | SHAP feature importance for LGBM            |

### Per-feature-set checkpoints (`outputs/ckpt/`)

For each feature set (AAC, DPC, TPC_k1000, SVD128), the pipeline writes:

| File                    | Content                                                      |
| ----------------------- | ------------------------------------------------------------ |
| `metrics_<fname>.csv`   | Aggregate metrics for that feature set (7 rows)              |
| `per_class_<fname>.csv` | Per-class metrics for that feature set                       |
| `bootstrap_<fname>.csv` | Bootstrap CI row for that feature set                        |
| `<fname>.done`          | Marker written last — signals "this feature set is complete" |

The `.done` marker is written last, after all CSVs are flushed. If the run crashes mid-feature-set, the marker is absent, and the next run reprocesses it.

### Cached probabilities (`outputs/proba/`)

For each base model × feature set, the pipeline saves the model's validation and test probabilities as:

```text
outputs/proba/<fname>_<model>_<hash>.npz
```

The `<hash>` is derived from the tuned hyperparameters, so changing params automatically invalidates the cache.

Each `.npz` contains:

```text
pva — validation probabilities, shape (n_val, n_classes)
pte — test probabilities, shape (n_test, n_classes)
pred — test predictions, shape (n_test,)
used — the hyperparameters used (JSON string)
```

On rerun, cached probabilities are loaded — no retraining.

### Feature matrix cache

Feature matrices are cached on local disk at:

```text
/content/cache/features/
```

This is the fast-but-ephemeral part of Colab — the cache is lost when the VM restarts, but rebuilding features is minutes, not hours.


## 12. Caching and Resumability

The pipeline is designed to survive crashes and to reuse work across runs.

### Feature-set checkpoints

After a feature set finishes:

```text
outputs/ckpt/<name>.done
```

is created.

On rerun, any feature set with a `.done` marker is skipped entirely.

### Per-model probability cache

Each model's validation and test probabilities are saved to:

```text
outputs/proba/<feature>_<model>_<hash>.npz
```

The hash is derived from the tuned hyperparameters, so changing params invalidates the cache automatically.

On rerun, cached probabilities are loaded — no retraining.

### Feature matrix cache

Feature matrices are cached on local disk at:

```text
/content/cache/features/
```

This is the fast-but-ephemeral part of Colab — the cache is lost when the VM restarts, but rebuilding from features is minutes, not hours.

### What gets skipped after a crash

| Component                         | Behavior            |
| --------------------------------- | ------------------- |
| Feature sets with `.done` markers | Fully skipped       |
| Models with `.npz` files          | Loaded, not refit   |
| Feature matrices cached           | Loaded, not rebuilt |

If TPC crashes mid-run, AAC and DPC are already saved and are skipped on the next run.

---

## 13. Interpretability

Two approaches are used.

### Feature importance (built-in)

Available from Random Forest (impurity-based) and LightGBM (gain-based).

Plots are saved as:

```text
importance_RF_AAC.png
importance_LGBM_AAC.png
```

### SHAP values (LightGBM only)

SHAP computes each feature's contribution to each prediction. The pipeline averages `|SHAP|` across samples and classes to produce a global feature importance ranking, saved as:

```text
shap_LGBM_AAC.png
```

The plot shows mean `|SHAP|` for the top 20 AAC features, sorted descending.

SHAP top features on AAC:

```text
W (tryptophan)
H (histidine)
A (alanine)
E (glutamate)
C (cysteine)
```

This is consistent with enzyme biochemistry — W is enriched in membrane-associated oxidoreductases; H is a common catalytic and metal-binding residue; C is a frequent nucleophile.

---

## 14. Results Summary

### Best macro F1 per feature set (test set)

| Feature set | Best base model | Best ensemble    |
| ----------- | --------------- | ---------------- |
| AAC         | RF — 0.744      | Hard — 0.744     |
| DPC         | LGBM — 0.894    | Stacking — 0.882 |
| TPC_k1000   | LGBM — 0.897    | Stacking — 0.883 |
| SVD128      | LGBM — 0.780    | Soft — 0.764     |

**Headline:** the highest single-model macro F1 is **0.897** (LGBM on TPC_k1000).

The highest ensemble macro F1 is **0.883** (Stacking on TPC_k1000).

### Observations

Gradient boosting (LGBM) outperforms bagging (RF) on the higher-dimensional feature sets by 5–15 points.

Sequence-order features (DPC, TPC_k1000) substantially outperform composition-only (AAC) and compression-only (SVD128).

Stacking does not beat the best single model. The differences are consistent with the base models being highly correlated (they share training data and features).

---

## 15. Limitations

### Random split, not homology-aware

Similar sequences can appear in both train and test. A clustered split (e.g., CD-HIT at 30% identity) would give lower, more honest numbers.

### Top-level EC only

The four-level hierarchy is collapsed to the first digit. Fine-grained subclass prediction is not attempted.

### Single-label only

Proteins annotated with multiple EC classes are reduced to the first class.

### SVM accuracy is below expectation

LinearSVM and RBF-SVM show macro F1 in the 0.2–0.57 range, far below RF and LGBM. The tuned hyperparameters were fit on 5K rows and do not transfer optimally to 208K-row training.

### No protein language model

A pretrained model (ESM-2, ProtBert) would likely outperform hand-crafted features.

### Feature importance shows association, not causation

A high-importance feature is predictive, not necessarily biologically causal.

---

## 16. Reproducibility

Random seed: **42**, set in `config.SEED` and used for all random operations.

Data splits: the dataset's predefined train/dev/test splits, unchanged.

Tuning: frozen in `cache/tuned_params.json`.

Environment: Python 3.9+, package versions in `requirements.txt`.

### To reproduce results exactly:

1. Use the same seed (`SEED = 42`).
2. Use the same dataset version (the Hugging Face hash at the time of the run).
3. Run Phase 1 (`tune_once.py`) with the same config, then Phase 2 (`pipeline.py`).
4. All metrics, plots, and CSVs will match.

To reproduce on a fresh machine, follow [§5](#5-how-to-run).
