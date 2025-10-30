# src/csp_custom.py
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

class CSP(BaseEstimator, TransformerMixin):
    """
    Simple CSP implementation.
    Find spatial filters maximizing variance difference between two classes.
    This version supports binary classification.
    """
    def __init__(self, n_components=4, reg=1e-6):
        self.n_components = n_components
        self.reg = reg

    def _cov(self, X):
        # X shape: (n_channels, n_times)
        X = X - X.mean(axis=1, keepdims=True)
        cov = X @ X.T / X.shape[1]
        return cov

    def fit(self, X, y):
        """
        X: array-like of shape (n_epochs, n_channels, n_times)
        y: labels (n_epochs,) expecting exactly 2 classes
        """
        classes = np.unique(y)
        if len(classes) != 2:
            raise ValueError("CSP implementation expects 2 classes.")
        covs = {c: 0 for c in classes}
        counts = {c: 0 for c in classes}
        for xi, yi in zip(X, y):
            cov = self._cov(xi)
            covs[yi] += cov
            counts[yi] += 1
        covs = {c:(covs[c]/counts[c]) for c in classes}
        # Composite covariance
        composite = covs[classes[0]] + covs[classes[1]]
        # eigen decomposition of composite
        evals, evecs = np.linalg.eigh(composite)
        # Whitening transform
        idx = np.argsort(evals)[::-1]
        evals = evals[idx]
        evecs = evecs[:, idx]
        P = np.diag(1.0/np.sqrt(evals + self.reg)) @ evecs.T
        # transform class covariances to whitened space
        S0 = P @ covs[classes[0]] @ P.T
        # eigen decomposition of S0
        e_vals0, e_vecs0 = np.linalg.eigh(S0)
        # sort
        idx0 = np.argsort(e_vals0)[::-1]
        e_vecs0 = e_vecs0[:, idx0]
        # filters in original space
        W = e_vecs0.T @ P
        # select components: first n and last n
        n = self.n_components // 2
        Wsel = np.vstack([W[:n, :], W[-n:, :]]) if self.n_components>0 else W
        self.filters_ = Wsel
        return self

    def transform(self, X):
        """
        Apply spatial filters and compute log-variance features.
        """
        feats = []
        for xi in X:
            # xi shape (n_channels, n_times)
            Xf = self.filters_ @ xi
            var = (Xf ** 2).mean(axis=1)
            feats.append(np.log(var / var.sum()))
        return np.asarray(feats)
