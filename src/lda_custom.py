# src/lda_custom.py
"""
Custom LDA (Linear Discriminant Analysis) classifier for 2-class BCI.
Implements the classifier from scratch using BaseEstimator/ClassifierMixin
for sklearn Pipeline compatibility.

Logical explanation:
- LDA finds the linear combination of features that best separates two classes.
- It computes the class means (μ₁, μ₂) and the pooled within-class covariance Sw.
- The projection vector is: w = Sw⁻¹ (μ₁ - μ₂)
- The decision boundary threshold is placed at the midpoint of the projected means.
- A new sample x is classified by projecting it: if w·x > threshold → class 1, else class 2.

This is the "Fisher's Linear Discriminant" — optimal under equal class covariance assumption.
"""

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin


class LDAClassifier(BaseEstimator, ClassifierMixin):
    """Custom 2-class LDA classifier from scratch."""

    def __init__(self, shrinkage=0.01):
        """
        Parameters
        ----------
        shrinkage : float
            Regularization parameter for the within-class scatter matrix.
            Sw_reg = (1 - shrinkage) * Sw + shrinkage * trace(Sw)/d * I
            Prevents singular matrix when n_samples < n_features.
        """
        self.shrinkage = shrinkage

    def fit(self, X, y):
        """
        Fit the LDA model.

        Steps:
        1. Compute class means μ₁, μ₂
        2. Compute within-class scatter Sw = Σ_c Σ_{x∈c} (x - μ_c)(x - μ_c)ᵀ
        3. Regularize: Sw_reg = (1-λ)·Sw + λ·(tr(Sw)/d)·I
        4. Projection vector: w = Sw_reg⁻¹ · (μ₁ - μ₂)
        5. Threshold: midpoint of projected class means
        """
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)

        self.classes_ = np.unique(y)
        if len(self.classes_) != 2:
            raise ValueError(f"LDAClassifier expects exactly 2 classes, got {len(self.classes_)}")

        c1, c2 = self.classes_
        X1 = X[y == c1]
        X2 = X[y == c2]

        # Step 1: class means
        self.mean1_ = np.mean(X1, axis=0)
        self.mean2_ = np.mean(X2, axis=0)

        # Step 2: within-class scatter matrix
        #   Sw = (X1 - μ1)ᵀ(X1 - μ1) + (X2 - μ2)ᵀ(X2 - μ2)
        d1 = X1 - self.mean1_
        d2 = X2 - self.mean2_
        Sw = d1.T @ d1 + d2.T @ d2
        Sw /= (len(X) - 2)  # unbiased estimate

        # Step 3: regularization (Ledoit-Wolf-style shrinkage)
        d = X.shape[1]
        trace_Sw = np.trace(Sw)
        Sw_reg = (1.0 - self.shrinkage) * Sw + self.shrinkage * (trace_Sw / d) * np.eye(d)

        # Step 4: projection vector w = Sw⁻¹ · (μ1 - μ2)
        diff = self.mean1_ - self.mean2_
        self.w_ = np.linalg.solve(Sw_reg, diff)

        # Step 5: threshold = midpoint of projected class means
        proj_mean1 = self.w_ @ self.mean1_
        proj_mean2 = self.w_ @ self.mean2_
        self.threshold_ = 0.5 * (proj_mean1 + proj_mean2)

        # Store which class has higher projection
        self.class_high_ = c1 if proj_mean1 > proj_mean2 else c2
        self.class_low_ = c2 if proj_mean1 > proj_mean2 else c1

        return self

    def decision_function(self, X):
        """Project X onto discriminant direction."""
        X = np.asarray(X, dtype=np.float64)
        return X @ self.w_ - self.threshold_

    def predict(self, X):
        """Classify each sample based on projection sign."""
        scores = self.decision_function(X)
        preds = np.where(scores > 0, self.class_high_, self.class_low_)
        return preds

    def predict_proba(self, X):
        """
        Approximate class probabilities using sigmoid of discriminant scores.
        This gives a smooth probability estimate compatible with sklearn scoring.
        """
        scores = self.decision_function(X)
        # Sigmoid: P(class_high) = 1 / (1 + exp(-score))
        prob_high = 1.0 / (1.0 + np.exp(-scores))
        prob_low = 1.0 - prob_high
        # Order columns to match self.classes_
        if self.class_high_ == self.classes_[0]:
            return np.column_stack([prob_high, prob_low])
        else:
            return np.column_stack([prob_low, prob_high])
