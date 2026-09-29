"""
End-to-end pipeline with:
  - Per-feature-set checkpointing (resumable after crashes)
  - Per-model probability caching (skips retraining)
  - Holdout stacking (no sklearn StackingClassifier)
  - In-place StandardScaler (one scaled copy)
  - Aggressive del + gc.collect between stages
  - Plots only for config.TARGET_FEATURE_FOR_PLOTS, only for RF, LGBM, LinearSVM
"""
import gc
import json
import os
import time
import hashlib
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from config import (
    OUT, CKPT, PROBA, FEAT_CACHE, SELECT_K, SEED,
    TARGET_FEATURE_FOR_PLOTS,
)
from src.data.loader import load_raw, clean, prepare
from src.features.aac import build_aac
from src.features.dpc import build_dpc
from src.features.tpc import build_tpc_sparse
from src.features.selectk import fit_selector
from src.features.svd_embed import fit_svd
from src.models.rf import tune_rf
from src.models.lgbm import tune_lgbm
from src.models.svm import tune_linear_svm, tune_rbf_svm
from src.ensembles.voting import combine_hard_vote, combine_soft_vote
from src.ensembles.stacking import holdout_stacking
from src.evaluation.metrics import full_report, bootstrap_metric_ci
from src.evaluation.plots import plot_confusion, plot_roc, plot_importance

from src.evaluation.checks import (
    check_no_duplicate_sequences, check_split_disjoint,
    check_class_distribution, check_no_nan_in_metrics,
)
from src.tuning.params import load_tuned_params
from src.explainability.shap_analysis import (
    explain_tree_model, plot_shap_summary,
    compare_with_builtin_importance,
)


# ------------------------------------------------------------------ #
# Utilities                                                           #
# ------------------------------------------------------------------ #
def step(msg):
    try:
        import psutil
        rss = psutil.Process().memory_info().rss / 1e9
        print(f"\n[{time.strftime('%H:%M:%S')}] === {msg} ===  (RAM {rss:.2f} GB)")
    except Exception:
        print(f"\n[{time.strftime('%H:%M:%S')}] === {msg} ===")


def feature_done(fname):
    return (CKPT / f"{fname}.done").exists()


def _atomic_csv(df, path):
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)


def _hash_params(params):
    return hashlib.md5(
        json.dumps(params, sort_keys=True, default=str).encode()
    ).hexdigest()[:8]


def should_plot(fname):
    return TARGET_FEATURE_FOR_PLOTS is not None and fname == TARGET_FEATURE_FOR_PLOTS


def record(results, per_class_rows, feature, model, params, rep):
    agg_row = {
        "feature": feature,
        "model":   model,
        "params":  json.dumps(params) if params else None,
    }
    for key in ["accuracy", "macro_precision", "macro_recall",
                "macro_f1", "weighted_f1", "mcc", "macro_auprc"]:
        agg_row[key] = rep.get(key)
    results.append(agg_row)

    pc = rep["per_class"].copy()
    pc.insert(0, "feature", feature)
    pc.insert(1, "model", model)
    pc = pc.reset_index()
    per_class_rows.append(pc)


# ------------------------------------------------------------------ #
# Base-model runner with probability caching                          #
# ------------------------------------------------------------------ #
def run_base(name, tuner, params, Xtr, y_tr, Xva, Xte, classes, fname):
    """Fit or load a base model.

    Returns (val_proba, test_proba, test_pred_int, used, model_or_None).
    Model is None on cache hit — caller must not rely on it for plots.
    """
    h = _hash_params(params)
    cache = PROBA / f"{fname}_{name}_{h}.npz"

    if cache.exists():
        z = np.load(cache, allow_pickle=True)
        used = json.loads(str(z["used"].item())) if "used" in z else params
        print(f"    [cache] loaded {name} {fname}")
        return (z["pva"].astype(np.float32),
                z["pte"].astype(np.float32),
                z["pred"].astype(np.int64),
                used,
                None)   # no model on cache hit

    model, _, used = tuner(Xtr, y_tr, params=params)

    # Class order must match — soft voting averages columns positionally.
    assert np.array_equal(model.classes_, np.asarray(classes)), \
        f"{name}: class order mismatch: {model.classes_} vs {classes}"

    pva = model.predict_proba(Xva).astype(np.float32)
    pte = model.predict_proba(Xte).astype(np.float32)
    pred = model.predict(Xte).astype(np.int32)   # int, not str

    np.savez(cache, pva=pva, pte=pte, pred=pred,
             used=json.dumps(used, default=str))
    print(f"    [cache] saved {name} {fname}")

    return pva, pte, pred, used, model


# ------------------------------------------------------------------ #
# Feature-level evaluation                                            #
# ------------------------------------------------------------------ #
def train_and_evaluate_one_feature_set(
    fname, Xtr, Xva, Xte,
    y_train_np, y_val_np, y_test_np, classes,
    rf_params, lgbm_params, lsvm_params, rsvm_params,
    results, per_class_rows, bootstrap_rows,
    test_df, rng,
):
    step(f"Feature set: {fname}")
    do_plots = should_plot(fname)

    # --- Materialize writable float32 arrays (mmap is read-only) ---
    Xtr = np.array(Xtr, dtype=np.float32, copy=True)
    Xva = np.array(Xva, dtype=np.float32, copy=True)
    Xte = np.array(Xte, dtype=np.float32, copy=True)

    # --- One scaled copy, standard scaler ---
    sc = StandardScaler()
    Xtr = sc.fit_transform(Xtr)
    Xva = sc.transform(Xva)
    Xte = sc.transform(Xte)

    # --- Base models ---
    P = {}
    for name, tuner, prm in [
        ("RF",        tune_rf,         rf_params),
        ("LGBM",      tune_lgbm,       lgbm_params),
        ("LinearSVM", tune_linear_svm, lsvm_params),
        ("RBF_SVM",   tune_rbf_svm,    rsvm_params),
    ]:
        t0 = time.time()
        pva, pte, pred, used, model = run_base(
            name, tuner, prm, Xtr, y_train_np, Xva, Xte, classes, fname,
        )
        rep = full_report(y_test_np, pred, pte, classes)
        record(results, per_class_rows, fname, name, used, rep)
        P[name] = (pva, pte, pred)
        print(f"  {name:9s} {fname}: macroF1={rep['macro_f1']:.4f}  "
              f"({time.time()-t0:.1f}s)")

        # ---- Plots for RF and LGBM only, on target feature set ----
        if do_plots and name in ("RF", "LGBM") and model is not None:
            try:
                plot_confusion(y_test_np, pred, classes, f"{name}_{fname}")
                plot_importance(model, [f"f{i}" for i in range(Xtr.shape[1])],
                                f"{name}_{fname}")
                print(f"    [plot] confusion + importance for {name} {fname}")
            except Exception as e:
                print(f"    [plot] skipped {name}: {e}")

            if name == "LGBM":
                try:
                    n = min(300, len(Xte))
                    idx = rng.choice(len(Xte), size=n, replace=False)
                    X_sample = np.asarray(Xte[idx]).copy()
                    feature_names = [f"f{i}" for i in range(Xte.shape[1])]
                    explainer, shap_vals = explain_tree_model(model, X_sample)
                    plot_shap_summary(
                        shap_vals, X_sample,
                        feature_names=feature_names,
                        save_path=OUT / f"shap_LGBM_{fname}.png",
                    )
                    cmp = compare_with_builtin_importance(
                        model, shap_vals, feature_names,
                    )
                    print(f"    [plot] SHAP for LGBM {fname} "
                          f"(top-20 overlap: {cmp.get('overlap_at_top_n','n/a')}/20)")
                    del explainer, shap_vals, X_sample
                    gc.collect()
                except Exception as e:
                    print(f"    [plot] SHAP skipped: {e}")

        del model
        gc.collect()

    # ---- ROC for LinearSVM on target feature set ----
    if do_plots:
        try:
            plot_roc(y_test_np, P["LinearSVM"][1], classes,
                     f"LinearSVM_{fname}")
            print(f"    [plot] ROC for LinearSVM {fname}")
        except Exception as e:
            print(f"    [plot] ROC skipped: {e}")

    # ---- Free features — probabilities remain ----
    del Xtr, Xva, Xte
    gc.collect()

    # ---- Hard Voting ----
    t0 = time.time()
    hard_pred = combine_hard_vote([P["RF"][2], P["LGBM"][2], P["LinearSVM"][2]])
    rep = full_report(y_test_np, hard_pred, None, classes)
    record(results, per_class_rows, fname, "HardVote", None, rep)
    print(f"  Hard      {fname}: macroF1={rep['macro_f1']:.4f}  "
          f"({time.time()-t0:.1f}s)")

    # ---- Soft Voting (RF + LGBM + RBF-SVM) ----
    t0 = time.time()
    soft_pred, soft_proba = combine_soft_vote(
        [P["RF"][1], P["LGBM"][1], P["RBF_SVM"][1]], classes
    )
    rep = full_report(y_test_np, soft_pred, soft_proba, classes)
    record(results, per_class_rows, fname, "SoftVote", None, rep)
    print(f"  Soft      {fname}: macroF1={rep['macro_f1']:.4f}  "
          f"({time.time()-t0:.1f}s)")

    # ---- Holdout Stacking (RF + LGBM + LinearSVM) ----
    t0 = time.time()
    stack_order = ["RF", "LGBM", "LinearSVM"]
    stack_pred, stack_proba = holdout_stacking(
        val_probas=[P[m][0] for m in stack_order],
        test_probas=[P[m][1] for m in stack_order],
        y_val=y_val_np,
        seed=SEED,
    )
    rep = full_report(y_test_np, stack_pred, stack_proba, classes)
    record(results, per_class_rows, fname, "Stacking", None, rep)
    print(f"  Stack     {fname}: macroF1={rep['macro_f1']:.4f}  "
          f"({time.time()-t0:.1f}s)")

    # ---- Bootstrap CI ----
    ci = bootstrap_metric_ci(y_test_np, stack_pred)
    bootstrap_rows.append({
        "feature": fname, "model": "Stacking",
        "point_estimate": ci["point_estimate"],
        "ci_low": ci["ci_low"], "ci_high": ci["ci_high"],
    })
    print(f"    [CI] Stacking {fname}: {ci['point_estimate']:.4f} "
          f"[{ci['ci_low']:.4f}, {ci['ci_high']:.4f}]")


# ------------------------------------------------------------------ #
# Checkpoint helpers                                                  #
# ------------------------------------------------------------------ #
EXPECTED_MODELS = ["RF", "LGBM", "LinearSVM", "RBF_SVM",
                   "HardVote", "SoftVote", "Stacking"]


def save_feature_checkpoint(fname, results, per_class_rows, bootstrap_rows):
    if len(results) < len(EXPECTED_MODELS):
        print(f"  [ckpt] INCOMPLETE — {len(results)}/{len(EXPECTED_MODELS)} rows "
              f"for {fname}; not marking done")
        _atomic_csv(pd.DataFrame(results), CKPT / f"metrics_{fname}.csv")
        return
    _atomic_csv(pd.DataFrame(results), CKPT / f"metrics_{fname}.csv")
    _atomic_csv(pd.concat(per_class_rows, ignore_index=True),
                CKPT / f"per_class_{fname}.csv")
    _atomic_csv(pd.DataFrame(bootstrap_rows),
                CKPT / f"bootstrap_{fname}.csv")
    (CKPT / f"{fname}.done").touch()
    print(f"  [ckpt] wrote {fname}.done ({len(results)} rows)")


def feature_path(name, split):
    return FEAT_CACHE / f"{name}_{split}.npy"


def feature_cached(name):
    return all(feature_path(name, s).exists() for s in ("train", "val", "test"))


def save_feature(name, Xtr, Xva, Xte):
    for split, X in (("train", Xtr), ("val", Xva), ("test", Xte)):
        np.save(feature_path(name, split), np.asarray(X, dtype=np.float32))


def load_feature(name):
    # Return as-is (may be mmap); the caller materializes.
    return tuple(
        np.load(feature_path(name, s))
        for s in ("train", "val", "test")
    )


# ------------------------------------------------------------------ #
# Main                                                                #
# ------------------------------------------------------------------ #
def run():
    rng = np.random.RandomState(SEED)

    step("Load tuned hyperparameters")
    tuned = load_tuned_params()
    if not tuned:
        raise RuntimeError("cache/tuned_params.json missing")
    rf_params   = tuned.get("RF")
    lgbm_params = tuned.get("LGBM")
    lsvm_params = tuned.get("LinearSVM")
    rsvm_params = tuned.get("RBF_SVM")
    for name, p in [("RF", rf_params), ("LGBM", lgbm_params),
                    ("LinearSVM", lsvm_params), ("RBF_SVM", rsvm_params)]:
        print(f"  {name:10s} {p}")

    step("Load data")
    train_raw, val_raw, test_raw = load_raw()
    train_df = prepare(clean(train_raw, verbose=False))
    val_df   = prepare(clean(val_raw,   verbose=False))
    test_df  = prepare(clean(test_raw,  verbose=False))

    print(f"  train: {len(train_df)}  val: {len(val_df)}  test: {len(test_df)}")

    classes = sorted(set(train_df["primary"]) | set(val_df["primary"])
                     | set(test_df["primary"]))
    print(f"  classes: {classes}")

    step("Data quality checks")
    check_no_duplicate_sequences(train_df, "train")
    check_no_duplicate_sequences(val_df,   "val")
    check_no_duplicate_sequences(test_df,  "test")
    check_split_disjoint(train_df, val_df, test_df)
    check_class_distribution(train_df, val_df, test_df, classes)

    train_seqs = train_df["seq"].tolist()
    val_seqs   = val_df["seq"].tolist()
    test_seqs  = test_df["seq"].tolist()
    y_train_np = train_df["primary"].values
    y_val_np   = val_df["primary"].values
    y_test_np  = test_df["primary"].values

    # --------------------------------------------------------------- #
    # 1. AAC                                                           #
    # --------------------------------------------------------------- #
    if feature_done("AAC"):
        print("\n[AAC] already done — skipping")
    else:
        step("Build features (AAC)")
        if feature_cached("AAC"):
            Xtr, Xva, Xte = load_feature("AAC")
            print("  AAC loaded from cache")
        else:
            t0 = time.time()
            Xtr = build_aac(train_seqs); Xva = build_aac(val_seqs); Xte = build_aac(test_seqs)
            save_feature("AAC", Xtr, Xva, Xte)
            print(f"  AAC built + cached in {time.time()-t0:.1f}s")
        results, pcr, boot = [], [], []
        train_and_evaluate_one_feature_set(
            "AAC", Xtr, Xva, Xte,
            y_train_np, y_val_np, y_test_np, classes,
            rf_params, lgbm_params, lsvm_params, rsvm_params,
            results, pcr, boot, test_df, rng,
        )
        save_feature_checkpoint("AAC", results, pcr, boot)
        del Xtr, Xva, Xte; gc.collect()

    # --------------------------------------------------------------- #
    # 2. DPC                                                           #
    # --------------------------------------------------------------- #
    if feature_done("DPC"):
        print("\n[DPC] already done — skipping")
    else:
        step("Build features (DPC)")
        if feature_cached("DPC"):
            Xtr, Xva, Xte = load_feature("DPC")
            print("  DPC loaded from cache")
        else:
            t0 = time.time()
            Xtr = build_dpc(train_seqs); Xva = build_dpc(val_seqs); Xte = build_dpc(test_seqs)
            save_feature("DPC", Xtr, Xva, Xte)
            print(f"  DPC built + cached in {time.time()-t0:.1f}s")
        results, pcr, boot = [], [], []
        train_and_evaluate_one_feature_set(
            "DPC", Xtr, Xva, Xte,
            y_train_np, y_val_np, y_test_np, classes,
            rf_params, lgbm_params, lsvm_params, rsvm_params,
            results, pcr, boot, test_df, rng,
        )
        save_feature_checkpoint("DPC", results, pcr, boot)
        del Xtr, Xva, Xte; gc.collect()

    # --------------------------------------------------------------- #
    # 3 & 4. TPC_k and SVD128 — need TPC raw                          #
    # --------------------------------------------------------------- #
    kname = f"TPC_k{SELECT_K}"
    needs_tpc = not feature_done(kname) or not feature_done("SVD128")

    if needs_tpc:
        del train_raw, val_raw, test_raw
        gc.collect()

        tpc_paths = [FEAT_CACHE / f"TPC_{s}.npz" for s in ("train","val","test")]
        if all(p.exists() for p in tpc_paths):
            print("\n[TPC raw] loaded from cache")
            import scipy.sparse
            X_tr_tpc = scipy.sparse.load_npz(tpc_paths[0])
            X_va_tpc = scipy.sparse.load_npz(tpc_paths[1])
            X_te_tpc = scipy.sparse.load_npz(tpc_paths[2])
        else:
            step("Build features (TPC raw)")
            t0 = time.time()
            X_tr_tpc = build_tpc_sparse(train_seqs)
            X_va_tpc = build_tpc_sparse(val_seqs)
            X_te_tpc = build_tpc_sparse(test_seqs)
            import scipy.sparse
            scipy.sparse.save_npz(tpc_paths[0], X_tr_tpc)
            scipy.sparse.save_npz(tpc_paths[1], X_va_tpc)
            scipy.sparse.save_npz(tpc_paths[2], X_te_tpc)
            print(f"  TPC raw built + cached in {time.time()-t0:.1f}s")

        # TPC_k
        if not feature_done(kname):
            step(f"Derive TPC_k{SELECT_K}")
            if feature_cached(kname):
                Xtr, Xva, Xte = load_feature(kname)
                print(f"  {kname} loaded from cache")
            else:
                t0 = time.time()
                sel = fit_selector(X_tr_tpc, y_train_np)
                Xtr = sel.transform(X_tr_tpc).toarray().astype(np.float32)
                Xva = sel.transform(X_va_tpc).toarray().astype(np.float32)
                Xte = sel.transform(X_te_tpc).toarray().astype(np.float32)
                save_feature(kname, Xtr, Xva, Xte)
                print(f"  {kname} built + cached in {time.time()-t0:.1f}s")
            results, pcr, boot = [], [], []
            train_and_evaluate_one_feature_set(
                kname, Xtr, Xva, Xte,
                y_train_np, y_val_np, y_test_np, classes,
                rf_params, lgbm_params, lsvm_params, rsvm_params,
                results, pcr, boot, test_df, rng,
            )
            save_feature_checkpoint(kname, results, pcr, boot)
            del Xtr, Xva, Xte; gc.collect()

        # SVD128
        if not feature_done("SVD128"):
            step("Derive SVD128")
            if feature_cached("SVD128"):
                Xtr, Xva, Xte = load_feature("SVD128")
                print("  SVD128 loaded from cache")
            else:
                t0 = time.time()
                svd = fit_svd(X_tr_tpc)
                Xtr = svd.transform(X_tr_tpc).astype(np.float32)
                Xva = svd.transform(X_va_tpc).astype(np.float32)
                Xte = svd.transform(X_te_tpc).astype(np.float32)
                save_feature("SVD128", Xtr, Xva, Xte)
                print(f"  SVD128 built + cached in {time.time()-t0:.1f}s")
            results, pcr, boot = [], [], []
            train_and_evaluate_one_feature_set(
                "SVD128", Xtr, Xva, Xte,
                y_train_np, y_val_np, y_test_np, classes,
                rf_params, lgbm_params, lsvm_params, rsvm_params,
                results, pcr, boot, test_df, rng,
            )
            save_feature_checkpoint("SVD128", results, pcr, boot)
            del Xtr, Xva, Xte; gc.collect()

        del X_tr_tpc, X_va_tpc, X_te_tpc
        gc.collect()

    # --------------------------------------------------------------- #
    # Assemble final metrics — only for the four expected feature sets #
    # --------------------------------------------------------------- #
    step("Assemble final metrics from checkpoints")
    feature_names = ["AAC", "DPC", f"TPC_k{SELECT_K}", "SVD128"]

    metrics_files = [CKPT / f"metrics_{fn}.csv"   for fn in feature_names
                     if (CKPT / f"metrics_{fn}.csv").exists()]
    per_class_files = [CKPT / f"per_class_{fn}.csv" for fn in feature_names
                       if (CKPT / f"per_class_{fn}.csv").exists()]
    boot_files = [CKPT / f"bootstrap_{fn}.csv"   for fn in feature_names
                  if (CKPT / f"bootstrap_{fn}.csv").exists()]

    if not metrics_files:
        raise RuntimeError("No feature-set metrics found — nothing to assemble")

    res_df = pd.concat([pd.read_csv(f) for f in metrics_files], ignore_index=True)
    pc_df  = pd.concat([pd.read_csv(f) for f in per_class_files], ignore_index=True) \
             if per_class_files else pd.DataFrame()
    bs_df  = pd.concat([pd.read_csv(f) for f in boot_files], ignore_index=True) \
             if boot_files else pd.DataFrame()

    preferred = ["feature", "model", "accuracy", "macro_precision",
                 "macro_recall", "macro_f1", "weighted_f1", "mcc",
                 "macro_auprc", "params"]
    cols = [c for c in preferred if c in res_df.columns]
    res_df = res_df[cols]

    res_df.to_csv(OUT / "metrics.csv", index=False)
    if not pc_df.empty:
        pc_df.to_csv(OUT / "per_class_all.csv", index=False)
    if not bs_df.empty:
        bs_df.to_csv(OUT / "bootstrap_cis.csv", index=False)

    print(res_df.to_string(index=False))

    step("Integrity checks")
    check_no_nan_in_metrics(res_df)
  

    return res_df


if __name__ == "__main__":
    run()
