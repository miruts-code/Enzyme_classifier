"""TruncatedSVD embedding from TPC counts."""
from sklearn.decomposition import TruncatedSVD
from config import SVD_DIM


def fit_svd(X_train, n_components=SVD_DIM, seed=42):
    svd = TruncatedSVD(n_components=n_components, random_state=seed)
    svd.fit(X_train)
    return svd
