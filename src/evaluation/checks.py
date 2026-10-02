"""Data quality, leakage, and integrity checks."""
import numpy as np
import pandas as pd


def check_no_duplicate_sequences(df, name):
    n_total = len(df)
    n_unique = df["seq"].nunique()
    n_dupes = n_total - n_unique
    
    if n_dupes == 0:
        print(f"[check:dupes] {name}: no duplicates ({n_total} rows)")
    else:
        print(f"[check:dupes] {name}: {n_dupes} duplicate rows "
                 f"({n_total} total, {n_unique} unique)")
    return n_dupes


def check_split_disjoint(train_df, val_df, test_df):
    train_seqs = set(train_df["seq"])
    val_seqs   = set(val_df["seq"])
    test_seqs  = set(test_df["seq"])
    train_val  = train_seqs & val_seqs
    train_test = train_seqs & test_seqs
    val_test   = val_seqs & test_seqs
    ok = not (train_val or train_test or val_test)
   
    print(f"[check:leakage] train ∩ val:  {len(train_val)} sequences")
    print(f"[check:leakage] train ∩ test: {len(train_test)} sequences")
    print(f"[check:leakage] val ∩ test:   {len(val_test)} sequences")
    print(f"[check:leakage] {'OK: splits are disjoint' if ok else 'WARNING: overlapping sequences'}")
    return ok


def check_class_distribution(train_df, val_df, test_df, classes):
    print("[check:balance] Class distribution per split:")
    print(f"  {'class':>6}  {'train':>15}  {'val':>15}  {'test':>15}")
    for c in classes:
        tr = (train_df["primary"] == c).sum()
        va = (val_df["primary"] == c).sum()
        te = (test_df["primary"] == c).sum()
        tr_pct = 100.0 * tr / len(train_df) if len(train_df) else 0
        va_pct = 100.0 * va / len(val_df) if len(val_df) else 0
        te_pct = 100.0 * te / len(test_df) if len(test_df) else 0
        print(f"  {c:>6}  {tr:>6} ({tr_pct:>4.1f}%)  "
              f"{va:>6} ({va_pct:>4.1f}%)  {te:>6} ({te_pct:>4.1f}%)")


def check_feature_fitted_on_train(selector, svd, X_train_shape, X_test_shape):
    try:
        tr_sel = selector.transform(np.zeros((1, X_train_shape[1]))).shape
        te_sel = selector.transform(np.zeros((1, X_test_shape[1]))).shape
        tr_svd = svd.transform(np.zeros((1, X_train_shape[1]))).shape
        te_svd = svd.transform(np.zeros((1, X_test_shape[1]))).shape
        ok = (tr_sel == te_sel) and (tr_svd == te_svd)
       
        print(f"[check:fit] selector shape: train={tr_sel}, test={te_sel}")
        print(f"[check:fit] svd shape:      train={tr_svd}, test={te_svd}")
        print(f"[check:fit] {'OK' if ok else 'MISMATCH'}")
        return ok
    except Exception as e:
        print(f"[check:fit] error: {e}")
        return False


def check_results_complete(results_df, expected_models, expected_features):
    expected = {(f, m) for f in expected_features for m in expected_models}
    actual   = set(zip(results_df["feature"], results_df["model"]))
    missing  = expected - actual
    extra    = actual - expected
    ok = not missing and not extra
   
    print(f"[check:results] expected {len(expected)} rows, found {len(actual)}")
    if missing:
        print(f"[check:results] MISSING: {missing}")
    if extra:
        print(f"[check:results] EXTRA: {extra}")
    if ok:
        print("[check:results] OK: all model x feature combinations present")
    return ok


def check_no_nan_in_metrics(results_df):
    metric_cols = ["accuracy", "macro_precision", "macro_recall",
                   "macro_f1", "weighted_f1", "mcc"]
    metric_cols = [c for c in metric_cols if c in results_df.columns]
    n_nan = int(results_df[metric_cols].isna().sum().sum())
    ok = n_nan == 0
 
    print(f"[check:nan] {'OK: no NaN in ' + str(len(metric_cols)) + ' metric columns' if ok else 'WARNING: ' + str(n_nan) + ' NaN values found'}")
    return ok


