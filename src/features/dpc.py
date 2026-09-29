"""Dipeptide Composition - 400-dim."""
import numpy as np
from config import VALID_AA

AA_LIST = sorted(VALID_AA)
AA_IDX = {aa: i for i, aa in enumerate(AA_LIST)}


def dpc_vector(seq):
    v = np.zeros(400, dtype=np.float32)
    n = len(seq) - 1
    if n <= 0:
        return v
    for i in range(n):
        a, b = seq[i], seq[i + 1]
        if a in AA_IDX and b in AA_IDX:
            v[AA_IDX[a] * 20 + AA_IDX[b]] += 1
    return v / n


def build_dpc(seqs):
    return np.vstack([dpc_vector(s) for s in seqs])
