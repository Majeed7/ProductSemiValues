import numpy as np

def esps_prefix(K):
    """
    Compute ESPs (elementary symmetric polynomials) for all prefixes of K.

    Vectorized across degrees (q) using slicing; only the outer loop over j remains.

    Parameters
    ----------
    K : array (d, m)
        Each row is a feature-wise kernel vector k_i of length m.

    Returns
    -------
    L : array (d+1, d+1, m)
        L[j, q] = degree-q ESP computed over the first j features (indices 0..j-1).
    """
    d, m = K.shape
    L = np.zeros((d + 1, d + 1, m), dtype=K.dtype)
    L[0, 0] = 1.0  # base: ESP of empty set = 1
    for j in range(1, d + 1):
        L[j, 0] = 1.0
        kj = K[j - 1]  # (m,)
        # For q = 1..j: L[j, q] = L[j-1, q] + kj * L[j-1, q-1]
        L[j, 1:j + 1] = L[j - 1, 1:j + 1] + kj * L[j - 1, 0:j]
    return L


def esps_suffix(K):
    """
    Compute ESPs for all suffixes of K.

    Vectorized across degrees using slicing; only the outer loop over j remains.

    Parameters
    ----------
    K : array (d, m)

    Returns
    -------
    R : array (d+1, d+1, m)
        R[j, q] = degree-q ESP computed over features j..(d-1).
    """
    d, m = K.shape
    R = np.zeros((d + 1, d + 1, m), dtype=K.dtype)
    R[d, 0] = 1.0  # base: ESP of empty set = 1
    for j in range(d - 1, -1, -1):
        R[j, 0] = 1.0
        kj = K[j]  # (m,)
        deg = d - j
        # For q = 1..deg: R[j, q] = R[j+1, q] + kj * R[j+1, q-1]
        R[j, 1:deg + 1] = R[j + 1, 1:deg + 1] + kj * R[j + 1, 0:deg]
    return R


def esps_minus_j_all(K, use_scaling=True, dtype=None):
    """
    Compute ESPs excluding each feature j, for all j.

    Uses stable prefix–suffix convolution instead of subtraction.

    Parameters
    ----------
    K : array (d, m)
        Kernel matrix (each row a feature vector).
    use_scaling : bool
        If True, normalize columns of K to improve stability.
    dtype : np.dtype or None
        Precision to use (np.float64 or np.longdouble).

    Returns
    -------
    e_minus : array (d, d, m)
        e_minus[j, q, :] = degree-q ESP excluding feature j.
    """
    if dtype is None:
        # use highest available precision
        dtype = np.longdouble if np.finfo(np.longdouble).precision > np.finfo(np.float64).precision else np.float64

    K = np.asarray(K, dtype=dtype)
    d, m = K.shape

    # Optional per-column scaling to reduce numerical blowup
    scales = np.ones(m, dtype=dtype)
    if use_scaling:
        scales = np.maximum(1.0, np.max(np.abs(K), axis=0))
        K = K / scales  # normalize per column

    # Compute prefix and suffix ESPs
    L = esps_prefix(K)  # ESPs of first j features
    R = esps_suffix(K)  # ESPs of last (d-j) features

    e_minus = np.zeros((d, d, m), dtype=dtype)
    for j in range(d):
        # For feature j, convolve prefix (0..j-1) and suffix (j+1..d-1)
        for q in range(d):
            tmin = max(0, q - (d - (j + 1)))  # ensure indices valid in R
            tmax = min(q, j)
            acc = np.zeros(m, dtype=dtype)
            for t in range(tmin, tmax + 1):
                acc += L[j, t] * R[j + 1, q - t]
            if use_scaling and np.any(scales != 1):
                acc *= scales ** q  # undo scaling
            e_minus[j, q] = acc
    return e_minus


def shapley_all_features_collective(K, alpha, mu, use_scaling=True, dtype=None):
    """
    Compute Shapley-like values for all features using the stable ESP method.

    Parameters
    ----------
    K : array (d, m)
        Kernel vectors per feature.
    alpha : array (m,)
        Weights for each column (same as in your original code).
    mu : array (d,)
        Precomputed Shapley coefficients for each degree q.
    use_scaling : bool
        Whether to normalize columns of K for stability.
    dtype : np.dtype or None
        Numerical precision.

    Returns
    -------
    shap : array (d,)
        Shapley value per feature.
    """
    K = np.asarray(K, dtype=(dtype or K.dtype))
    d, m = K.shape
    mu = np.asarray(mu, dtype=K.dtype)
    assert mu.shape[0] >= d, "mu must have at least d entries for q=0..d-1"

    # Compute all e^{(-j)} once
    e_minus = esps_minus_j_all(K, use_scaling=use_scaling, dtype=dtype)
    ones = np.ones(m, dtype=K.dtype)
    shap = np.zeros(d, dtype=K.dtype)

    for j in range(d):
        # Combine ESPs with Shapley weights
        result = (mu[:d].reshape(d, 1) * e_minus[j]).sum(axis=0)
        # Shapley aggregation step
        shap[j] = alpha.dot((K[j] - ones) * result)
    return shap
