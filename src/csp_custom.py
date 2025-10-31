import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from scipy.linalg import eigh

class CSP(BaseEstimator, TransformerMixin):
    """CSP (Common Spatial Pattern) implementation for 2-class EEG."""

    def __init__(self, n_components=4, reg=1e-10):
        self.n_components = n_components
        self.reg = reg
        self.filters_ = None
        self.n_channels = None

    def fit(self, X, y):
        print(f"[DEBUG] CSP.fit X shape: {X.shape}")
        X = np.asarray(X)
        y = np.asarray(y)

        classes = np.unique(y)
        if len(classes) != 2:
            raise ValueError("CSP implementation expects exactly 2 classes.")
        
        self.classes_ = classes
        self.n_channels = X.shape[1]

        covs = []
        for c in classes:
            Xc = X[y == c]
            cov = np.zeros((self.n_channels, self.n_channels))
            for xi in Xc:
                xi = np.asarray(xi)
                if xi.ndim == 1:
                    xi = xi.reshape(self.n_channels, -1)
                elif xi.shape[0] != self.n_channels and xi.shape[1] == self.n_channels:
                    xi = xi.T
                cov_i = np.dot(xi, xi.T) / xi.shape[1]
                cov_i /= np.trace(cov_i)
                cov += cov_i
            cov /= len(Xc)
            covs.append(cov)

        cov_a, cov_b = covs
        eigvals, eigvecs = eigh(cov_a, cov_a + cov_b + self.reg * np.eye(self.n_channels))

        ix = np.argsort(eigvals)[::-1]
        eigvecs = eigvecs[:, ix]

        n_half = self.n_components // 2
        self.filters_ = np.hstack([eigvecs[:, :n_half], eigvecs[:, -n_half:]])

        print(f"✅ CSP fitted. Filters shape: {self.filters_.shape}")
        return self

    def transform(self, X):
        """Transformation des signaux via les filtres CSP."""
        X = np.asarray(X)
        feats = []

        for i, xi in enumerate(X):
            xi = np.asarray(xi)

            # ✅ S'assure que xi est bien 2D : (n_channels, n_times)
            if xi.ndim == 1:
                # Un seul canal ? -> on le considère comme (1, n_times)
                xi = xi.reshape(self.n_channels, -1)
            elif xi.shape[0] != self.n_channels and xi.shape[1] == self.n_channels:
                xi = xi.T
            elif xi.shape[0] != self.n_channels:
                print(f"[⚠️ CSP.transform] Epoch {i} skipped: bad shape {xi.shape}")
                continue

            # ✅ Application des filtres spatiaux
            Xf = np.dot(self.filters_.T, xi)

            # ✅ Calcul des log-variances de manière stable
            var = np.var(Xf, axis=1)
            var = np.maximum(var, 1e-12)
            feats.append(np.log(var / np.sum(var)))

        feats = np.array(feats)
        print(f"[DEBUG] CSP.transform output shape: {feats.shape}")
        return feats

