# src/eigen_custom.py
"""
Custom eigenvalue / eigenvector decomposition for symmetric (Hermitian) matrices.
This is the "harder bonus" of the Total Perspective Vortex project.

Algorithm: QR iteration with Wilkinson shifts
----------------------------------------------
For a real symmetric matrix A:
1. Reduce A to tridiagonal form T using Householder reflections: A = Q₀ T Q₀ᵀ
2. Apply QR iteration with implicit Wilkinson shifts on T:
   - At each step, factor (T - σI) = QR, then T ← RQ + σI
   - Eigenvalues converge on the diagonal
3. Accumulate eigenvectors: V = Q₀ · Q₁ · Q₂ · ...

This converges cubically for symmetric matrices with Wilkinson shifts.

Also implements generalized eigenvalue: A·v = λ·B·v
via Cholesky decomposition of B and reduction to standard form.
"""

import numpy as np


def _householder_tridiag(A):
    """
    Reduce symmetric matrix A to tridiagonal form using Householder reflections.
    Returns (T, Q) where A = Q T Qᵀ, T is tridiagonal, Q is orthogonal.
    
    Algorithm:
    For each column k (0 to n-3):
      - Take the sub-diagonal part of column k: x = A[k+1:, k]
      - Compute Householder vector v such that Hx = ||x||·e₁
      - Apply H = I - 2vvᵀ/vᵀv to both sides: A ← HAH
    """
    n = A.shape[0]
    T = A.astype(np.float64).copy()
    Q = np.eye(n, dtype=np.float64)

    for k in range(n - 2):
        x = T[k + 1:, k].copy()
        alpha = -np.sign(x[0]) * np.linalg.norm(x)
        if abs(alpha) < 1e-15:
            continue

        # Householder vector
        v = x.copy()
        v[0] -= alpha
        v = v / np.linalg.norm(v)

        # Apply reflection to T: T ← (I - 2vvᵀ) T (I - 2vvᵀ)
        # Efficiently: T[k+1:, :] -= 2v (vᵀ T[k+1:, :])
        sub = T[k + 1:, :]
        sub -= 2.0 * np.outer(v, v @ sub)
        T[:, k + 1:] = T[:, k + 1:] - 2.0 * (T[:, k + 1:] @ v).reshape(-1, 1) @ v.reshape(1, -1)

        # Accumulate: Q ← Q (I - 2vvᵀ)
        Qsub = Q[:, k + 1:]
        Qsub -= 2.0 * (Qsub @ v).reshape(-1, 1) @ v.reshape(1, -1)

    return T, Q


def _wilkinson_shift(a, b, c):
    """
    Compute Wilkinson shift for the trailing 2×2 block:
    [[a, b],
     [b, c]]
    Returns the eigenvalue of this 2×2 closer to c (bottom-right).
    """
    delta = (a - c) / 2.0
    if abs(delta) < 1e-15:
        return c - abs(b)
    sign_d = 1.0 if delta >= 0 else -1.0
    return c - b ** 2 / (delta + sign_d * np.sqrt(delta ** 2 + b ** 2))


def _qr_iteration_tridiag(T, Q, max_iter=1000, tol=1e-12):
    """
    Implicit QR iteration with Wilkinson shifts on tridiagonal matrix T.
    Returns (eigenvalues, eigenvectors).
    
    The off-diagonal elements converge to zero, leaving eigenvalues on the diagonal.
    """
    n = T.shape[0]
    T = T.astype(np.float64).copy()
    V = Q.astype(np.float64).copy()

    for iteration in range(max_iter):
        # Check convergence: find the largest off-diagonal element
        off_diag_max = 0.0
        for i in range(n - 1):
            off_diag_max = max(off_diag_max, abs(T[i, i + 1]))
        if off_diag_max < tol:
            break

        # Find the lowest unconverged block
        m = n - 1
        while m > 0 and abs(T[m, m - 1]) < tol:
            m -= 1
        if m == 0:
            break

        # Find top of unreduced block
        l = m - 1
        while l > 0 and abs(T[l, l - 1]) >= tol:
            l -= 1

        # Wilkinson shift from trailing 2×2
        sigma = _wilkinson_shift(T[m - 1, m - 1], T[m - 1, m], T[m, m])

        # Shifted QR step on the block [l:m+1, l:m+1]
        # Using Givens rotations for efficiency on tridiagonal
        x = T[l, l] - sigma
        z = T[l + 1, l]

        for k in range(l, m):
            # Compute Givens rotation to zero out z
            r = np.sqrt(x * x + z * z)
            if r < 1e-15:
                break
            c = x / r
            s = -z / r

            # Apply Givens rotation to T: G(k, k+1) from left and right
            # Rows k, k+1
            for j in range(max(0, k - 1), min(n, k + 3)):
                t1 = T[k, j]
                t2 = T[k + 1, j]
                T[k, j] = c * t1 - s * t2
                T[k + 1, j] = s * t1 + c * t2

            # Columns k, k+1
            for j in range(max(0, k - 1), min(n, k + 3)):
                t1 = T[j, k]
                t2 = T[j, k + 1]
                T[j, k] = c * t1 - s * t2
                T[j, k + 1] = s * t1 + c * t2

            # Accumulate eigenvectors
            for j in range(n):
                v1 = V[j, k]
                v2 = V[j, k + 1]
                V[j, k] = c * v1 - s * v2
                V[j, k + 1] = s * v1 + c * v2

            # Prepare for next Givens rotation
            if k < m - 1:
                x = T[k + 1, k]
                z = T[k + 2, k]

    eigenvalues = np.diag(T)
    return eigenvalues, V


def eigh_custom(A):
    """
    Compute eigenvalues and eigenvectors of a real symmetric matrix A.
    
    Returns (eigenvalues, eigenvectors) sorted by ascending eigenvalue,
    same interface as scipy.linalg.eigh.
    
    Algorithm:
    1. Tridiagonalize A using Householder reflections
    2. Apply QR iteration with Wilkinson shifts
    """
    A = np.asarray(A, dtype=np.float64)
    n = A.shape[0]

    if n == 1:
        return np.array([A[0, 0]]), np.array([[1.0]])

    if n == 2:
        # Direct formula for 2×2 symmetric
        a, b, d = A[0, 0], A[0, 1], A[1, 1]
        trace = a + d
        det = a * d - b * b
        disc = np.sqrt(max(0, trace * trace - 4 * det))
        l1 = (trace - disc) / 2
        l2 = (trace + disc) / 2
        if abs(b) < 1e-15:
            return np.array([l1, l2]), np.eye(2)
        v1 = np.array([b, l1 - a])
        v2 = np.array([b, l2 - a])
        v1 /= np.linalg.norm(v1)
        v2 /= np.linalg.norm(v2)
        return np.array([l1, l2]), np.column_stack([v1, v2])

    # Make symmetric
    A = 0.5 * (A + A.T)

    # Step 1: Householder tridiagonalization
    T, Q = _householder_tridiag(A)

    # Step 2: QR iteration
    eigenvalues, eigenvectors = _qr_iteration_tridiag(T, Q)

    # Sort by ascending eigenvalue
    idx = np.argsort(eigenvalues)
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    return eigenvalues, eigenvectors


def eigh_generalized_custom(A, B):
    """
    Solve the generalized eigenvalue problem: A v = λ B v
    
    Method: Cholesky factorization of B.
      B = L Lᵀ  →  L⁻¹ A L⁻ᵀ u = λ u  →  v = L⁻ᵀ u
    
    Falls back to regularized B if Cholesky fails.
    """
    A = np.asarray(A, dtype=np.float64)
    B = np.asarray(B, dtype=np.float64)

    # Regularize B to ensure positive definite
    reg = 1e-10
    B_reg = B + reg * np.eye(B.shape[0])

    try:
        L = np.linalg.cholesky(B_reg)
    except np.linalg.LinAlgError:
        # Heavier regularization
        B_reg = B + 1e-6 * np.trace(B) / B.shape[0] * np.eye(B.shape[0])
        L = np.linalg.cholesky(B_reg)

    # Transform: C = L⁻¹ A L⁻ᵀ
    L_inv = np.linalg.inv(L)
    C = L_inv @ A @ L_inv.T
    C = 0.5 * (C + C.T)  # enforce symmetry

    eigenvalues, U = eigh_custom(C)

    # Back-transform eigenvectors: v = L⁻ᵀ u
    eigenvectors = np.linalg.solve(L.T, U)

    return eigenvalues, eigenvectors
