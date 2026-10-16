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

### Aggregate results ( AAC and DPC)

| Feature | Model | Accuracy | Macro Precision | Macro Recall | Macro F1 | Weighted F1 | MCC | Macro AUPRC |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| AAC | RF | 0.7404 | 0.7768 | 0.7271 | 0.7439 | 0.7411 | 0.6658 | 0.8208 |
| AAC | LGBM | 0.7182 | 0.7167 | 0.7474 | 0.7296 | 0.7192 | 0.6454 | 0.8229 |
| AAC | LinearSVM | 0.4001 | 0.4079 | 0.2258 | 0.2186 | 0.3102 | 0.1434 | 0.2887 |
| AAC | RBF_SVM | 0.4396 | 0.4279 | 0.4128 | 0.4169 | 0.4414 | 0.2849 | 0.3942 |
| AAC | HardVote | 0.7354 | 0.7994 | 0.7118 | 0.7438 | 0.7346 | 0.6563 | — |
| AAC | SoftVote | 0.7323 | 0.7823 | 0.7154 | 0.7417 | 0.7310 | 0.6518 | 0.8141 |
| AAC | Stacking | 0.7454 | 0.7081 | 0.7856 | 0.7394 | 0.7473 | 0.6844 | 0.8631 |
| DPC | RF | 0.8138 | 0.9044 | 0.7739 | 0.8228 | 0.8123 | 0.7601 | 0.9066 |
| DPC | LGBM | 0.8777 | 0.9076 | 0.8821 | 0.8940 | 0.8782 | 0.8424 | 0.9523 |
| DPC | LinearSVM | 0.5400 | 0.5889 | 0.4715 | 0.5049 | 0.5236 | 0.3853 | 0.5490 |
| DPC | RBF_SVM | 0.5192 | 0.6537 | 0.4188 | 0.4721 | 0.5045 | 0.3506 | 0.5298 |
| DPC | HardVote | 0.8301 | 0.9012 | 0.7990 | 0.8410 | 0.8298 | 0.7800 | — |
| DPC | SoftVote | 0.8585 | 0.9080 | 0.8492 | 0.8752 | 0.8586 | 0.8167 | 0.9324 |
| DPC | Stacking | 0.8777 | 0.8648 | 0.9027 | 0.8824 | 0.8780 | 0.8452 | 0.9605 |                                                                                                                           

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

On rerun, cached probabilities are loaded — no retraining.


## 12. Caching and Resumability

The pipeline is designed to survive crashes and to reuse work across runs.

### Feature-set checkpoints

After a feature set finishes:

```text
outputs/ckpt/<name>.done
```

is created.

On rerun, any feature set with a `.done` marker is skipped entirely.

## 13. Interpretability

### Feature importance 

Available from Random Forest (impurity-based) 

<img width="1170" height="910" alt="importance_RF_AAC" src="https://github.com/user-attachments/assets/e5a3d0b1-ba91-4d25-97ad-b29403155214" />

### SHAP values (LightGBM )

SHAP computes each feature's contribution to each prediction. The pipeline averages `|SHAP|` across samples and classes to produce a global feature importance ranking.

<img width="1184" height="881" alt="shap_LGBM_AAC" src="https://github.com/user-attachments/assets/40e8e13d-a251-49fa-aa2e-7965e3524da2" />
The plot shows mean `|SHAP|` for the top 20 AAC features, sorted descending.

### Confusion MatriX

confusion matrix compares a model's predicted classes with the actual classes to show correct and incorrect classification to each class.

<img width="1170" height="910" alt="confusion_LGBM_AAC" src="https://github.com/user-attachments/assets/2f14367e-0896-4699-9dc9-daa91a710784" />


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

- **Sequence-order information improves classification.** LightGBM’s macro F1 increases from 0.730 with AAC to 0.894 with DPC, while TPC_k1000 yields a smaller improvement to 0.897. This suggests that dipeptide frequencies capture substantial class-discriminative information, with diminishing returns from additional sequence-order features.

- **Dimensionality reduction may discard useful information.** SVD128 achieves a macro F1 of 0.780, below the 0.897 obtained with TPC_k1000. This suggests that the compressed representation may not preserve all class-relevant patterns captured by the selected tripeptide features.

- **Model performance depends on feature representation.** LightGBM outperforms Random Forest on DPC and TPC_k1000, while Random Forest performs slightly better on AAC. This difference may reflect the models’ varying ability to capture informative patterns in sparse, high-dimensional features versus dense composition features.

- **Ensembling provides no consistent improvement.** The best ensemble achieves a macro F1 of 0.883, below LightGBM’s 0.897 on TPC_k1000. Overlapping information among base models and weaker SVM predictions may limit the benefits of combining them.

- **LinearSVM struggles to distinguish several classes.** Its macro F1 of 0.219 on AAC and low recall for multiple classes suggest that a linear decision boundary may not capture the relationships needed for effective classification. hyperparameter selection may also contribute.

- **RBF-SVM shows limited performance with sparse, high-dimensional features.** Trained on only 2,000 sequences, it achieves relatively low macro F1 scores. Limited training coverage may contribute to poor recall for minority classes, although further experiments are needed to confirm this explanation.
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
