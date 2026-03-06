import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from scipy.linalg import eigh as scipy_eigh

# Import custom eigenvalue decomposition (bonus)
try:
    from src.eigen_custom import eigh_generalized_custom, eigh_custom
except ImportError:
    eigh_generalized_custom = None
    eigh_custom = None


class CSP(BaseEstimator, TransformerMixin):
    """CSP (Common Spatial Pattern) implementation for 2-class EEG.
    
    Can use either scipy or custom eigenvalue decomposition (bonus).
    Set use_custom_eigen=True to use the from-scratch implementation.
    """

    def __init__(self, n_components=4, reg=1e-10, shrink=0.1, use_custom_eigen=False):
        self.n_components = n_components
        self.reg = reg
        self.shrink = shrink
        self.use_custom_eigen = use_custom_eigen
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
                # covariance normalized by trace
                cov_i = np.dot(xi, xi.T) / xi.shape[1]
                cov_i /= np.trace(cov_i) if np.trace(cov_i) != 0 else 1.0
                # shrinkage: mélange vers une matrice diagonale (utile si peu d'exemples)
                if self.shrink and 0.0 < self.shrink < 1.0:
                    avg = np.trace(cov_i) / self.n_channels
                    cov_i = (1 - self.shrink) * cov_i + self.shrink * (avg * np.eye(self.n_channels))
                cov += cov_i
            # moyenne sur les epochs de la classe
            if len(Xc) > 0:
                cov /= len(Xc)
            covs.append(cov)   

        cov_a, cov_b = covs
        # generalized eigenvalue problem
        B = cov_a + cov_b + self.reg * np.eye(self.n_channels)

        if self.use_custom_eigen and eigh_generalized_custom is not None:
            print("[CSP] Using CUSTOM eigenvalue decomposition (bonus)")
            eigvals, eigvecs = eigh_generalized_custom(cov_a, B)
        else:
            eigvals, eigvecs = scipy_eigh(cov_a, B)

        ix = np.argsort(eigvals)[::-1]
        eigvecs = eigvecs[:, ix]

        # assure exactement n_components choisis (balance left/right)
        n_comp = min(self.n_components, self.n_channels)
        n_left = int(np.ceil(n_comp / 2))
        n_right = int(np.floor(n_comp / 2))

        parts = []
        if n_left > 0:
            parts.append(eigvecs[:, :n_left])
        if n_right > 0:
            parts.append(eigvecs[:, -n_right:])
        if parts:
            self.filters_ = np.hstack(parts)
        else:
            self.filters_ = eigvecs[:, :n_comp]

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

