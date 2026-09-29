"""Amino Acid Composition - 20-dim."""
import numpy as np
from config import VALID_AA

AA_LIST = sorted(VALID_AA)
AA_IDX = {aa: i for i, aa in enumerate(AA_LIST)}


def aac_vector(seq):
    v = np.zeros(20, dtype=np.float32)
    for aa in seq:
        if aa in AA_IDX:
            v[AA_IDX[aa]] += 1
    n = len(seq)
    return v / n if n else v


def build_aac(seqs):
    return np.vstack([aac_vector(s) for s in seqs])
