"""Load SwissProt-EC using the dataset's default train/dev/test splits."""
import ast
import re
import numpy as np
import pandas as pd
from datasets import load_dataset
from config import HF_DATASET, VALID_AA, EC_CLASS_NAMES

# Match EC:<digit>.  — accepts partial annotations like EC:6.-.-.-
EC_RE = re.compile(r"EC:(\d+)\.")


def load_raw():
    """Return (train_df, val_df, test_df) from the dataset's own splits."""
    train_ds = load_dataset(HF_DATASET, split="train")
    val_ds   = load_dataset(HF_DATASET, split="dev")
    test_ds  = load_dataset(HF_DATASET, split="test")
    return (
        train_ds.to_pandas(),
        val_ds.to_pandas(),
        test_ds.to_pandas(),
    )


def _parse_labels(value):
    """Return a list of label strings from either a list or a stringified list."""
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        try:
            parsed = ast.literal_eval(value)
            if isinstance(parsed, list):
                return parsed
        except (ValueError, SyntaxError):
            pass
    return []


def extract_classes(labels_str_list):
    """Return sorted list of top-level EC classes present in the row.

    Accepts:
      - a real list of strings: ['EC:6.-.-.-', ...]
      - a stringified list:     "['EC:6.-.-.-', ...]"
    """
    labels = _parse_labels(labels_str_list)
    classes = set()
    for s in labels:
        m = EC_RE.match(str(s))
        if m:
            classes.add(int(m.group(1)))
    return sorted(classes)


def clean(df, verbose=True):
    n0 = len(df)
    mask_valid = df["seq"].apply(lambda s: all(aa in VALID_AA for aa in s))
    df = df[mask_valid].copy()
    df["classes"] = df["labels_str"].apply(extract_classes)
    df = df[df["classes"].map(len) > 0].copy()
    n1 = len(df)
    if verbose:
        print(f"[clean] {n0} -> {n1} rows  (dropped {n0-n1})")
        ec7 = df["classes"].apply(lambda c: 7 in c).sum()
        if ec7:
            print(f"[warn] {ec7} rows contain EC 7 (Translocases)")
    return df


def prepare(df):
    df = df.reset_index(drop=True)
    df["primary"] = df["classes"].apply(lambda c: c[0])
    return df


def human_readable(label):
    return EC_CLASS_NAMES.get(int(label), f"EC {label}")
