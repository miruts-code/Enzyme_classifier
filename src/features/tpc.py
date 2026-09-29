"""Tripeptide Composition — 8000-dim (sparse)."""
import numpy as np
from scipy.sparse import lil_matrix
from config import VALID_AA

AA_LIST = sorted(VALID_AA)
AA_IDX = {aa: i for i, aa in enumerate(AA_LIST)}


def tpc_row(seq):
    n = len(seq) - 2
    if n <= 0:
        return {}
    counts = {}
    for i in range(n):
        a, b, c = seq[i], seq[i + 1], seq[i + 2]
        if a in AA_IDX and b in AA_IDX and c in AA_IDX:
            key = (AA_IDX[a] * 20 + AA_IDX[b]) * 20 + AA_IDX[c]
            counts[key] = counts.get(key, 0) + 1
    return {k: v / n for k, v in counts.items()}


def build_tpc_sparse(seqs):
    X = lil_matrix((len(seqs), 8000), dtype=np.float32)
    for i, s in enumerate(seqs):
        for k, v in tpc_row(s).items():
            X[i, k] = v
    return X.tocsr()
