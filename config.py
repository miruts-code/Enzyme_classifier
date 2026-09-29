"""Global configuration."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CACHE = ROOT / "cache"
OUT = ROOT / "outputs"
CACHE.mkdir(exist_ok=True)
OUT.mkdir(exist_ok=True)

# Cache directories
CKPT = OUT / "ckpt"
PROBA = OUT / "proba"
FEAT_CACHE = Path("/content/cache/features")
for d in (CKPT, PROBA, FEAT_CACHE):
    d.mkdir(parents=True, exist_ok=True)

HF_DATASET = "DanielHesslow/SwissProt-EC"
SEED = 42

AAC_DIM = 20
DPC_DIM = 400
TPC_DIM = 8000
SVD_DIM = 128

SELECT_K = 1000
TRIPEPTIDE_K_GRID = [100, 300, 500, 1000]

TUNING_SUBSAMPLE_SIZE = 5000
N_ITER_RANDOM_SEARCH = 10
CV_FOLDS = 3
SCORING = "f1_macro"

SVM_SUBSAMPLE = 2000
RBF_SVM_PROBABILITY = True

N_BOOTSTRAP = 300

# Plots generated only for this feature set and only for RF, LGBM, LinearSVM.
TARGET_FEATURE_FOR_PLOTS = "AAC"

EC_CLASS_NAMES = {
    1: "Oxidoreductases",
    2: "Transferases",
    3: "Hydrolases",
    4: "Lyases",
    5: "Isomerases",
    6: "Ligases",
    7: "Translocases",
}
CLASS_NAMES = EC_CLASS_NAMES
CLASS_LABELS = sorted(EC_CLASS_NAMES.keys())

VALID_AA = set("ACDEFGHIKLMNPQRSTVWY")
