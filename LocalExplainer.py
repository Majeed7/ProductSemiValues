import numpy as np
from functools import reduce
from sklearn.metrics.pairwise import rbf_kernel
import math

import numpy as np
from numba import njit, prange

import numpy as np
from numba import njit, prange

import numpy as np
import jax
import jax.numpy as jnp
from jax import lax


'''
it solves the Mobius representation of Shapley values in RKHS with product kernel with Gauss Legendre quadrature
'''

## Numpy version

def weighted_values_gauss_shared(K, alpha, m_q=None):
    """
    Approximate ∑_m K[i,m] * alpha[m] * ∫_0^1 ∏_{j≠i} (1 + x*K[j,m]) dx  for all i,
    using shared Gauss nodes and log-space products (no ESPs, no prefix/suffix).
    K: (d, m)  kernel vectors (k_j) per feature j and column m
    alpha: (m,)
    Exact for degree (d-1) if m_q >= ceil(d/2).
    """
    K = np.asarray(K, dtype=np.float64)
    alpha = np.asarray(alpha, dtype=np.float64)
    d, m = K.shape
    if m_q is None:
        m_q = (d + 1) // 2  # ceil(d/2)

    # Quadrature nodes/weights on [0,1]
    x, w = np.polynomial.legendre.leggauss(m_q)  # [-1,1]
    x = 0.5*(x + 1.0)
    w = 0.5*w


    # --- Shared product across features PER node&column, in log-space ---
    # We stream over features to avoid a (m_q,d,m) tensor in memory.
    log_abs_P = np.zeros((m_q, m), dtype=np.float64)
    sign_P = np.ones((m_q, m), dtype=np.float64)
    for j in range(d):
        t = 1.0 + np.outer(x, K[j, :])            # shape (m_q, m)
        sign_P *= np.sign(t)
        log_abs_P += np.log(np.abs(t), dtype=np.float64)

    # --- Per-i division and integration ---
    # result[i] = sum_m K[i,m] * alpha[m] * sum_ell w[ell] * (P / (1 + x*K[i,m]))
    result = np.zeros(d, dtype=np.float64)
    wa = w[:, None]                               # (m_q,1)
    a = alpha[None, :]                            # (1,m)
    for i in range(d):
        denom = 1.0 + np.outer(x, K[i, :])        # (m_q, m)
        integrand_sign = sign_P * np.sign(denom)
        integrand_log = log_abs_P - np.log(np.abs(denom), dtype=np.float64)
        Qint = (wa * (integrand_sign * np.exp(integrand_log))).sum(axis=0)  # (m,)
        result[i] = (K[i, :] * a * Qint).sum()
    
    return result

def weighted_values_gauss_legendre(K, alpha, m_q=None):
    """
    Exact for degree (d-1) polynomials if m_q >= ceil(d/2).
    K: (d, m)  kernel vectors (k_j)
    alpha: (m,)
    """
    K = np.asarray(K, dtype=np.float64)
    alpha = np.asarray(alpha, dtype=np.float64)
    d, m = K.shape
    if m_q is None:
        m_q = (d + 1) // 2  # ceil(d/2)

    # Gauss-Legendre nodes/weights on [-1,1], map to [0,1]
    x, w = np.polynomial.legendre.leggauss(m_q)   # exact for deg <= 2*m_q-1 on [-1,1]
    x = 0.5 * (x + 1.0)                           # to [0,1]
    w = 0.5 * w

    acc_vec = np.zeros((d, m), dtype=K.dtype)     # accumulate ∫ Q_i(x) dx elementwise
    # shapes: K (d,m), x (m_q,), w (m_q,)
    X = x[:, None, None]                  # (m_q,1,1)
    B = 1.0 + X * K[None, :, :]           # (m_q, d, m)

    # prefix/suffix per node independently (axis=1 is features)
    pref = np.cumprod(B, axis=1)
    pref = np.concatenate(
        [np.ones((m_q,1,m), B.dtype), pref[:, :-1, :]],
        axis=1
    )

    suf = np.cumprod(B[:, ::-1, :], axis=1)[:, ::-1, :]
    suf = np.concatenate(
        [suf[:, 1:, :], np.ones((m_q,1,m), B.dtype)],
        axis=1
    )

    Q_no_i = pref * suf                   # (m_q, d, m)
    acc_vec = (w[:, None, None] * Q_no_i).sum(axis=0)  # (d, m)

    # vector version
    # for t in range(m_q):
    #     xt = x[t]; wt = w[t]
    #     B = 1.0 + xt * K                          # (d, m)

    #     # exclusive product across features (no division), per quadrature node
    #     pref = np.cumprod(B, axis=0)
    #     pref = np.vstack([np.ones((1, m)), pref[:-1]])
    #     suf  = np.cumprod(B[::-1], axis=0)[::-1]
    #     suf  = np.vstack([suf[1:], np.ones((1, m))])

    #     Q_no_i = pref * suf                       # (d, m): ∏_{j≠i}(1+xt*k_j)
    #     acc_vec += wt * Q_no_i

    # Now S_i' = alpha^T (k_i * acc_vec_i)
    return (K * acc_vec * alpha[None, :]).sum(axis=1)

'''
JAX version
'''
@jax.jit
def _weighted_values_gl_core(K, alpha, x, w):
    """
    Core JAX kernel.
    K: (d,m)        alpha: (m,)
    x: (m_q,) nodes in [0,1]   w: (m_q,) weights
    returns: (d,)
    """
    B = 1.0 + x[:, None, None] * K[None, :, :]      # (m_q,d,m)

    
    # exclusive prefix
    pref = lax.cumprod(B, axis=1)
    pref = jnp.concatenate(
        [jnp.ones((B.shape[0], 1, B.shape[2]), dtype=B.dtype), pref[:, :-1, :]],
        axis=1
    )
    # exclusive suffix
    suf = lax.cumprod(B[:, ::-1, :], axis=1)[:, ::-1, :]
    suf = jnp.concatenate(
        [suf[:, 1:, :], jnp.ones((B.shape[0], 1, B.shape[2]), dtype=B.dtype)],
        axis=1
    )

    Q = pref * suf                                  # (m_q,d,m)
    acc = (w[:, None, None] * Q).sum(axis=0)        # (d,m)
    return (K * acc * alpha[None, :]).sum(axis=1)   # (d,)

def weighted_values_gl_jax_auto(K: np.ndarray, alpha: np.ndarray, m_q=None):
    """
    One-shot function: builds Gauss–Legendre nodes/weights (mapped to [0,1])
    and returns the weighted values.

    K: (d,m) numpy array
    alpha: (m,) numpy array
    m_q: number of Gauss–Legendre nodes
    returns: (d,) numpy array
    """

    if m_q == None:
        m_q = (K.shape[0] + 1) // 2   # ceil(d/2)
    
    # 1) Build nodes/weights on [-1,1], then map to [0,1]
    x_np, w_np = np.polynomial.legendre.leggauss(m_q)
    x_np = 0.5 * (x_np + 1.0)   # to [0,1]
    w_np = 0.5 * w_np

    # 2) Respect input dtype/device preference
    dtype = np.result_type(K.dtype, alpha.dtype, np.float32)
    backend_platforms = {dev.platform for dev in jax.devices()}
    if (jax.default_backend() != "cpu" or backend_platforms & {"gpu", "metal", "tpu"}
            or (np.issubdtype(dtype, np.floating) and dtype.itemsize > 4)):
        dtype = np.float32
        K = K.astype(np.float32, copy=False)
        alpha = alpha.astype(np.float32, copy=False)
    x_np = x_np.astype(dtype, copy=False)
    w_np = w_np.astype(dtype, copy=False)

    x = jnp.asarray(x_np, dtype=dtype)
    w = jnp.asarray(w_np, dtype=dtype)
    Kj = jnp.asarray(K, dtype=dtype)
    alphaj = jnp.asarray(alpha, dtype=dtype)

    # 3) Call the jitted core
    out = _weighted_values_gl_core(Kj, alphaj, x, w)

    # 4) Return to numpy (or keep as JAX array if you prefer)
    return np.asarray(out)


@jax.jit
def _weighted_values_gauss_shared_logscan_core(K, alpha, x, w):
    """
    JAX core mirroring weighted_values_gauss_shared via log-space prefix/suffix scans.
    Eliminates Python loops by scanning over feature axis.

    Shapes:
      K: (d, m), alpha: (m,), x: (m_q,), w: (m_q,)
    Returns (d,)
    """
    # Build B[ell, i, m] = 1 + x[ell] * K[i, m]
    B = 1.0 + x[:, None, None] * K[None, :, :]        # (m_q, d, m)

    # Sign/log-abs for stability
    sign_B = jnp.sign(B)
    log_abs_B = jnp.log(jnp.abs(B))

    # Inclusive prefix over features; convert to exclusive by shifting
    pref_log_incl = jnp.cumsum(log_abs_B, axis=1)
    pref_sign_incl = lax.cumprod(sign_B, axis=1)
    zeros_log = jnp.zeros((B.shape[0], 1, B.shape[2]), dtype=B.dtype)
    ones_sign = jnp.ones((B.shape[0], 1, B.shape[2]), dtype=B.dtype)
    pref_log_excl = jnp.concatenate([zeros_log, pref_log_incl[:, :-1, :]], axis=1)
    pref_sign_excl = jnp.concatenate([ones_sign, pref_sign_incl[:, :-1, :]], axis=1)

    # Inclusive suffix via reverse scan; then exclusive by shifting
    suf_log_incl = jnp.cumsum(log_abs_B[:, ::-1, :], axis=1)[:, ::-1, :]
    suf_sign_incl = lax.cumprod(sign_B[:, ::-1, :], axis=1)[:, ::-1, :]
    suf_log_excl = jnp.concatenate([suf_log_incl[:, 1:, :], zeros_log], axis=1)
    suf_sign_excl = jnp.concatenate([suf_sign_incl[:, 1:, :], ones_sign], axis=1)

    # Product excluding feature i
    log_abs_Q = pref_log_excl + suf_log_excl
    sign_Q = pref_sign_excl * suf_sign_excl
    Q = sign_Q * jnp.exp(log_abs_Q)                   # (m_q, d, m)

    # Integrate over nodes and reduce over m with alpha
    acc = (w[:, None, None] * Q).sum(axis=0)          # (d, m)
    return (K * acc * alpha[None, :]).sum(axis=1)


def weighted_values_gauss_shared_jax(K: np.ndarray, alpha: np.ndarray, m_q=None):
    """
    JAX version of weighted_values_gauss_shared.
    - Generates Gauss–Legendre nodes/weights internally and maps to [0,1].
    - Uses jitted core with prefix/suffix scans in log-space (no Python loops).

    Parameters
    ----------
    K : (d, m) numpy array
    alpha : (m,) numpy array
    m_q : int or None (defaults to ceil(d/2))

    Returns
    -------
    (d,) numpy array
    """
    d = K.shape[0]
    if m_q is None:
        m_q = (d + 1) // 2

    # Build nodes/weights on [-1,1] and map to [0,1]
    x_np, w_np = np.polynomial.legendre.leggauss(m_q)
    x_np = 0.5 * (x_np + 1.0)
    w_np = 0.5 * w_np

    # To JAX, preserve sensible dtype
    dtype = np.result_type(K.dtype, alpha.dtype, np.float32)
    x = x_np.copy() #jnp.asarray(x_np, dtype=dtype)
    w = jnp.asarray(w_np, dtype=dtype)
    Kj = jnp.asarray(K, dtype=dtype)
    alphaj = jnp.asarray(alpha, dtype=dtype)

    out = _weighted_values_gauss_shared_logscan_core(Kj, alphaj, x, w)
    return np.asarray(out)


def unweighted_values_from_kernel_vectors(kernel_vectors, alpha):
    """
    kernel_vectors: array of shape (d, m), each row is k_j (vector for feature j)
    alpha: array of shape (m,)
    Returns: weighted_value per feature, shape (d,)
    """
    K = np.asarray(kernel_vectors)       # shape (d, m)
    d, m = K.shape
    b = 1.0 + K                          # (d, m)
    
    # Compute global product across features (axis=0) → shape (m,)
    P = b.prod(axis=0)
    
    # Exclusive products: remove each feature
    # Avoid division by zero → use prefix/suffix products
    pref = np.cumprod(b, axis=0)         # prefix inclusive
    pref = np.vstack([np.ones((1, m)), pref[:-1]])   # make exclusive
    suf = np.cumprod(b[::-1], axis=0)[::-1]          # suffix inclusive reversed
    suf = np.vstack([suf[1:], np.ones((1, m))])      # make exclusive
    
    E = pref * suf   # E[i,:] = product of all b[j,:], j != i
    
    # Finally: S[i] = alpha dot (a_i * E[i])
    result = (K * E * alpha[None, :]).sum(axis=1)
    return result


# ---------- Main Explainer Classes ----------

class ProductKernelLocalExplainer:
    def __init__(self, model):
        """
        Initialize the Shapley Value Explainer.

        Args:
            model: A scikit-learn model (GP, SVM or SVR) with RBF kernel
        """
        self.model = model
        self.X_train = self.get_X_train()
        self.alpha = self.get_alpha()
        self.n, self.d = self.X_train.shape

    def get_X_train(self):
        """
        Retrieve the training sample  based on the model type.

        Returns:
            2D-array of samples.
        """
        if hasattr(self.model, "support_vectors_"):  # For SVM/SVR
            return self.model.support_vectors_

        if hasattr(self.model, "X_fit_"):  # For KRR
            return self.model.X_fit_
        
        elif hasattr(self.model, "X_train_"):  # For GP
            return self.model.X_train_

        if hasattr(self.model, "base_estimator_") and hasattr(self.model.base_estimator_, "X_train_"):  # for GP classifier
            return self.model.base_estimator_.X_train_

        else:
            raise ValueError("Unsupported model type for Shapley value computation.")
        
    def get_alpha(self):
        """
        Retrieve the alpha values based on the model type.

        Returns:
            Array of alpha values required for Shapley value computation.
        """
        if hasattr(self.model, "dual_coef_"):  # For SVM/SVR
            self.null_game = self.model.intercept_
        
            return self.model.dual_coef_.flatten()
        
        elif hasattr(self.model, "alpha_"):  # For GP
            alpha = self.model.alpha_.flatten()
            self.null_game = np.sum(alpha)
            
            return alpha
        
        else:
            raise ValueError("Unsupported model type for Shapley value computation.")
    
    def precompute_mu(self, d):
        """
        Precompute mu coefficients (as in the paper) or the weights in Shapley values.

        Args:
            d: Number of features.

        Returns:
            List of precomputed mu coefficients.
        """

        unnormalized_factors = [(math.factorial(q) * math.factorial(d - q - 1)) for q in range(d)]

        return np.array(unnormalized_factors) / math.factorial(d) 
    
    def compute_elementary_symmetric_polynomials_recursive(self, kernel_vectors):
        """
        Compute elementary symmetric polynomials.

        Args:
            kernel_vectors: List of kernel vectors (1D arrays) of features 
                (for local explainer, it is computed by realizing kernel function between each feature of x (instance under explanation) and training set).

        Returns:
            e: List of elementary symmetric polynomials .
        """
    
        # Compute power sums
        s = [
            sum([np.power(k, p) for k in kernel_vectors])
            for p in range(0, len(kernel_vectors) + 1)
        ]
        
        # Compute elementary symmetric polynomials
        e = [np.ones_like(kernel_vectors[0])]  # e_0 = 1
        
        for r in range(1, len(kernel_vectors) + 1):
            term = 0 
            for k in range(1, r + 1):
                term += ((-1) ** (k-1)) * e[r - k] * s[k]
            e.append(term / r )
        
        return e

    def compute_elementary_symmetric_polynomials(self, kernel_vectors, scale=False):
        """
        Compute elementary symmetric polynomials using a dynamic programming approach.

        Args:
            kernel_vectors: List of kernel vectors (1D arrays).

        Returns:
            elementary: List of elementary symmetric polynomials.
        """
        # Initialize with e_0 = 1
        max_abs_k = 1.0
        if scale:
            max_abs_k = max(np.max(np.abs(k)) for k in kernel_vectors) or 1.0

        scaled_kernel = [k / max_abs_k for k in kernel_vectors]

        # Initialize polynomial coefficients: P_0(x) = 1
        e = [np.ones_like(scaled_kernel[0], dtype=np.float64)]

        for k in scaled_kernel:
            # Prepend and append zeros to handle polynomial multiplication (x - k)
            new_e = [np.zeros_like(e[0])] * (len(e) + 1)
            # new_e[0] corresponds to the constant term after multiplying by (x - k)
            new_e[0] = -k * e[0]
            # Compute the rest of the terms
            for i in range(1, len(e)):
                new_e[i] = e[i-1] - k * e[i]
            # The highest degree term is x^{len(e)}, coefficient is e[-1] (which is 1 initially)
            new_e[len(e)] = e[-1].copy()
            e = new_e
        
        # Extract elementary symmetric polynomials from the coefficients
        n = len(scaled_kernel)
        elementary = [np.ones_like(e[0])]  # e_0 = 1
        for r in range(1, n + 1):
            sign = (-1) ** r
            elementary_r = sign * e[n - r] * (max_abs_k ** r)
            elementary.append(elementary_r)
        
        return elementary
    
    def explain_by_kernel_vectors(self, kernel_vectors):
        """
        Compute Shapley values for all features of an instance based on computed feature-wise kernel vectors

        Args:
            kernel_vectors: feature-wise kernel vectors between x and training samples

        Returns:
            List of Shapley values, one for each feature.
        """

        shapley_values = []
        for j in range(self.d):
            shapley_values.append(self._compute_shapley_value(kernel_vectors, j))
        
        return shapley_values

    def _compute_shapley_value(self, kernel_vectors, feature_index):
        """
        Compute the Shapley value for a specific feature of an instance.

        Args:
            X: The dataset (2D array) used to train the model.
            x: The instance (1D array) for which to compute the Shapley value.
            feature_index: Index of the feature.

        Returns:
            Shapley value for the specified feature.
        """

        cZ_minus_j = [kernel_vectors[i] for i in range(self.d) if i != feature_index]
        e_polynomials = self.compute_elementary_symmetric_polynomials(cZ_minus_j)
        mu_coefficients = self.precompute_mu(self.d)
        
        # Compute kernel vector for the chosen feature
        k_j = kernel_vectors[feature_index]
        onevec = np.ones_like(k_j)
        
        # Compute the Shapley value
        result = np.zeros_like(k_j)
        for q in range(self.d):
            if q < len(e_polynomials):
                result += mu_coefficients[q] * e_polynomials[q]
        
        shapley_value = self.alpha.dot((k_j - onevec) * result)
        return shapley_value.item()
    
    def v_S(self, kernel_vectors, S):
        """
        Compute v(S): the inner product of alpha with the elementwise product of kernel_vectors columns in S.

        Args:
            kernel_vectors: list or np.ndarray of shape (d, n) or (n, d), kernel values for each feature and training point.
            S: iterable of indices (features to include).

        Returns:
            Scalar value: alpha^T (elementwise product of columns in S).
        """
        # Ensure kernel_vectors is (n, d)
        if isinstance(kernel_vectors, list):
            kernel_vectors = np.array(kernel_vectors).T  # shape (n, d)
        elif kernel_vectors.shape[0] != self.n:
            kernel_vectors = kernel_vectors.T  # shape (n, d)

        if len(S) == 0:
            prod = np.ones(self.n)
        else:
            prod = np.prod(kernel_vectors[:, list(S)], axis=1)
        return np.dot(self.alpha, prod)

    def brute_force_shapley(self, kernel_vectors):
        """
        Brute-force computation of Shapley values for all features using the Mobius representation.

        Args:
            kernel_vectors: np.ndarray of shape (n, d), kernel values for each training point and feature.
            alpha: np.ndarray of shape (n,), model coefficients.

        Returns:
            np.ndarray of Shapley values for all features (shape: d,).
        """
        import itertools
        n, d = kernel_vectors.shape
        shapley_values = np.zeros(d)

        features = list(range(d))
        # Iterate over all subsets S of features
        for subset_size in range(1, d + 1):
            for S in itertools.combinations(features, subset_size):
                # Compute m(S)
                prod_S = np.ones(n)
                for idx in S:
                    prod_S *= (kernel_vectors[:, idx] - 1)
                m_S = np.dot(self.alpha, prod_S)
                # Add contribution to all phi_i for i in S
                for i in S:
                    shapley_values[i] += (1.0 / subset_size) * m_S

        return shapley_values
    
class RBFLocalExplainer(ProductKernelLocalExplainer):
    
    def __init__(self, model):
        """
        Initialize the Shapley Value Explainer.

        Args:
            model: A scikit-learn model (GP, SVM or SVR) with RBF kernel
        """
        super().__init__(model)
        self.gamma = self.get_gamma()

    def get_gamma(self):
        """
        Retrieve the gamma parameter based on the model type.

        Returns:
            Gamma parameter for the RBF kernel.
        """
        if hasattr(self.model, "_gamma"):  # For SVM/SVR
            return self.model._gamma
        
        if hasattr(self.model, "gamma"):  # For KRR
            if self.model.gamma is not None:
                return self.model.gamma
            elif self.model.get_params()['kernel'] == 'rbf':
                return 1.0 / self.model.X_fit_.shape[1]

        elif hasattr(self.model.kernel_, "length_scale"):  # For GP (kernel_ has the posterior kernel fitted to the data)
            return (2 * (self.model.kernel_.length_scale ** 2)) ** -1
        

        else:
            raise ValueError("Unsupported model type for Shapley value computation.")

    def compute_kernel_vectors(self, X, x):
        """
        Compute kernel vectors for a given dataset X and instance x.

        Args:
            X: The dataset (2D array) used to train the model.
            x: The instance (1D array) for which to compute Shapley values.

        Returns:
            List of kernel vectors corresponding to each feature. Length = number of features.
        """

        # Initialize the kernel matrix
        kernel_vectors = []

        # For each sample and each feature, compute k(x_i^j, x^j)
        for i in range(self.d):
            kernel_vec = rbf_kernel(X[:,i].reshape(-1,1), x[...,np.newaxis][i].reshape(1,-1), gamma=self.gamma)
            kernel_vectors.append(kernel_vec.squeeze())    

        return kernel_vectors

    def _compute_shapley_value(self, kernel_vectors, feature_index, type='individual'):
        """
        Compute the Shapley value for a specific feature of an instance.

        Args:
            X: The dataset (2D array) used to train the model.
            x: The instance (1D array) for which to compute the Shapley value.
            feature_index: Index of the feature.

        Returns:
            Shapley value for the specified feature.
        """
        
        alpha = self.alpha 
        if type == 'individual':
            cZ_minus_j = [kernel_vectors[i] for i in range(self.d) if i != feature_index]
            e_polynomials = self.compute_elementary_symmetric_polynomials(cZ_minus_j)
            mu_coefficients = self.precompute_mu(self.d)
            
            # Compute kernel vector for the chosen feature
            k_j = kernel_vectors[feature_index]
            onevec = np.ones_like(k_j)
            
            # Compute the Shapley value
            result = np.zeros_like(k_j)
            for q in range(self.d):
                if q < len(e_polynomials):
                    result += mu_coefficients[q] * e_polynomials[q]

        elif type == 'collective':
            # Compute ESPs once over all features, then derive ESPs excluding j via synthetic division
            K = np.asarray(kernel_vectors)
            d, m = K.shape
            mu_coefficients = self.precompute_mu(self.d)

            # Global ESPs E[0..d], E[q] has shape (m,)
            E = np.zeros((d + 1, m), dtype=K.dtype)
            E[0] = 1.0
            for j in range(d):
                for q in range(j + 1, 0, -1):
                    E[q] += K[j] * E[q - 1]
            # E = self.compute_elementary_symmetric_polynomials(kernel_vectors)
            # For this feature j, compute e^{(-j)} via recurrence:
            # e^{(-j)}_0 = 1, e^{(-j)}_q = E[q] - k_j * e^{(-j)}_{q-1}
            k_j = K[feature_index]
            onevec = np.ones_like(k_j)

            q_prev = onevec.copy()                 # e^{(-j)}_0
            result = mu_coefficients[0] * q_prev   # accumulate mu[0]*e^{(-j)}_0
            for q in range(1, d):
                q_curr = E[q] - k_j * q_prev
                result += mu_coefficients[q] * q_curr
                q_prev = q_curr
        else:
            raise ValueError("type must be either 'individual' or 'collective'")
        
        shapley_value = alpha.dot((k_j - onevec) * result)

        
        return shapley_value.item()
    
    def explain(self, x):
        """
        Compute Shapley values for all features of an instance.

        Args:
            X: The dataset (2D array) used to train the model.
            x: The instance (1D array) for which to compute Shapley values.

        Returns:
            List of Shapley values, one for each feature.
        """
        
        import time
        
        kernel_vectors = self.compute_kernel_vectors(self.X_train, x)
        
        start_time = time.time()
        shapley_values = []
        for j in range(self.d):
             shapley_values.append(self._compute_shapley_value(kernel_vectors, j))
        print(f"Time taken to compute Shapley value: {time.time() - start_time} seconds")

        start_time = time.time()
        from esp_stable import shapley_all_features_collective
        shap = shapley_all_features_collective(np.array(kernel_vectors), self.alpha, self.precompute_mu(self.d), use_scaling=False)
        print(f"Time taken for collective Shapley value: {time.time() - start_time} seconds")
        
        ## this is the fastest method
        start_time = time.time()
        shap_gausleg_jax = weighted_values_gauss_shared_jax(np.asarray(kernel_vectors)-1, self.alpha, 100)
        print(f"Time taken for Gauss-Legendre Shapley value: {time.time() - start_time} seconds")

        start_time = time.time()
        shap_gausleg = weighted_values_gl_jax_auto(np.asarray(kernel_vectors)-1, self.alpha, 100)
        print(f"Time taken for Gauss-Legendre Shapley value: {time.time() - start_time} seconds")


        start_time = time.time()
        shap_gausleg = weighted_values_gauss_shared(np.asarray(kernel_vectors)-1, self.alpha, 100)
        print(f"Time taken for Gauss-Legendre Shapley value: {time.time() - start_time} seconds")

        start_time = time.time()
        shap_gausleg = weighted_values_gauss_legendre(np.asarray(kernel_vectors)-1, self.alpha, len(x) // 2)
        print(f"Time taken for Gauss-Legendre Shapley value: {time.time() - start_time} seconds")

        # start_time = time.time()
        # shap_gauslejax = weighted_values_gl_jax_auto(np.asarray(kernel_vectors)-1, self.alpha, 50)
        # print(f"Time taken for Gauss-Legendre Shapley value: {time.time() - start_time} seconds")

        E2 = self.compute_elementary_symmetric_polynomials(kernel_vectors)
        start_time = time.time()
        E, shap_parallel = weighted_values_from_kernel_vectors(np.asarray(kernel_vectors), self.precompute_mu(self.d), self.alpha, np.array(E2))
        print(f"Time taken for parallel Shapley value: {time.time() - start_time} seconds")


        start_time = time.time()
        banzhaf_vals = unweighted_values_from_kernel_vectors(np.asarray(kernel_vectors), self.alpha)
        print(f"Time taken for parallel Banzhaf value: {time.time() - start_time} seconds")



        return shapley_values
    
    def explain_brute_force(self, x):
        """
        Compute Shapley values for all features of an instance using brute-force method.

        Args:
            kernel_vectors: np.ndarray of shape (n, d), kernel values for each training point and feature.

        Returns:
            List of Shapley values, one for each feature.
        """
        
        kernel_vectors = self.compute_kernel_vectors(self.X_train, x)
        return self.brute_force_shapley(np.array(kernel_vectors).T)


# Example Usage
if __name__ == "__main__":
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, ConstantKernel
    from sklearn.datasets import make_classification, make_regression
    from sklearn.model_selection import train_test_split
    from sklearn.svm import SVC, SVR
    from sklearn.kernel_ridge import KernelRidge
    from sklearn.preprocessing import StandardScaler

    # Generate a synthetic regression dataset with 10 features
    X, y = make_regression(n_samples=600, n_features=10000, random_state=42)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

    # Standardize the features
    # scaler = StandardScaler()
    # X_train = scaler.fit_transform(X_train)
    # X_test = scaler.transform(X_test)

    # # Train an SVR model with RBF kernel
    # svr_model = SVR(kernel='rbf', C=1.0, gamma='scale')
    # svr_model.fit(X_train, y_train)

    # # train a GP
    kernel =  RBF(1.0, (1e-3, 1e3))
    model = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, alpha=1e-2)
    model.fit(X_train, y_train)


    # Initialize the explainer with this model
    explainer = RBFLocalExplainer(model)

    # Test instance
    x = X_test[0]  # Instance to explain

    # Compute Shapley values
    shapley_values = explainer.explain(x)
    print("Shapley Values:", shapley_values)


    # Train a Kernel Ridge Regression model with RBF kernel
    krr_model = KernelRidge(kernel='rbf', alpha=1.0, gamma=0.1)
    krr_model.fit(X_train, y_train)

    # Initialize the explainer with the KRR model
    explainer_krr = RBFLocalExplainer(krr_model)

    # Test instance
    x_krr = X_test[0]

    # Compute Shapley values for KRR
    shapley_values_krr = explainer_krr.explain(x_krr)
    print("Shapley Values (KRR):", shapley_values_krr)
    print(f"sum of Shapley values (KRR): {sum(shapley_values_krr)}")
    print("Predicted value (KRR):", krr_model.predict([x_krr])[0])

    # shap_vals = explainer.explain_brute_force(x)

    # print(f"sum of Shapley values is: {sum(shapley_values)}")

        
        # ------------------------- Classification Example -------------------------
    from sklearn.datasets import make_classification
    from sklearn.svm import SVC
    from sklearn.gaussian_process import GaussianProcessClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    
    # Generate a synthetic classification dataset (binary classification)
    X_clf, y_clf = make_classification(n_samples=200, n_features=10000, n_informative=5, 
                                    n_redundant=2, n_classes=2, random_state=42)  # <sup data-citation="6" className="inline select-none [&>a]:rounded-2xl [&>a]:border [&>a]:px-1.5 [&>a]:py-0.5 [&>a]:transition-colors shadow [&>a]:bg-ds-bg-subtle [&>a]:text-xs [&>svg]:w-4 [&>svg]:h-4 relative -top-[2px] citation-shimmer"><a href="https://machinelearningmastery.com/gaussian-processes-for-classification-with-python/">6</a></sup>

    # Split the data into training and testing sets
    X_train_clf, X_test_clf, y_train_clf, y_test_clf = train_test_split(X_clf, y_clf, test_size=0.3, random_state=42)

    # Standardize the features
    scaler_clf = StandardScaler()
    X_train_clf = scaler_clf.fit_transform(X_train_clf)
    X_test_clf = scaler_clf.transform(X_test_clf)

    # Train an SVC model with RBF kernel (set probability=True to enable probability estimates)
    svc_model = SVC(kernel='rbf', C=1.0, gamma='scale', probability=False, random_state=42)
    svc_model.fit(X_train_clf, y_train_clf)  # <sup data-citation="4" className="inline select-none [&>a]:rounded-2xl [&>a]:border [&>a]:px-1.5 [&>a]:py-0.5 [&>a]:transition-colors shadow [&>a]:bg-ds-bg-subtle [&>a]:text-xs [&>svg]:w-4 [&>svg]:h-4 relative -top-[2px] citation-shimmer"><a href="https://www.geeksforgeeks.org/support-vector-regression-svr-using-linear-and-non-linear-kernels-in-scikit-learn/">4</a></sup>

    # Initialize the explainer with the chosen classifier (e.g., the GP classifier)
    explainer_clf = RBFLocalExplainer(svc_model)  # use same explainer interface as for regression

    # Test instance for classification
    x_clf = X_test_clf[0]  # instance to explain

    # Compute Shapley values for classification
    shapley_values_clf = explainer_clf.explain(x_clf)
    print("Shapley Values (Classification):", shapley_values_clf)

    # You can also observe the predicted probability and the predicted class:
    print(f"sum of Shapley vlaue {sum(shapley_values_clf)}")
    print("predicted decision function: ", svc_model.decision_function([x_clf])[0])
    print("intercept is: ", svc_model.intercept_)



    # Alternatively, train a Gaussian Process Classifier with an RBF kernel
    kernel = RBF(1.0, (1e-3, 1e3))
    gpc = GaussianProcessClassifier(kernel=kernel, n_restarts_optimizer=10, random_state=42)
    gpc.fit(X_train_clf, y_train_clf)  # 

    # Initialize the explainer with the chosen classifier (e.g., the GP classifier)
    explainer_clf = RBFLocalExplainer(gpc)  # use same explainer interface as for regression

    # Test instance for classification
    x_clf = X_test_clf[0]  # instance to explain

    # Compute Shapley values for classification
    shapley_values_clf = explainer_clf.explain(x_clf)
    print("Shapley Values (Classification):", shapley_values_clf)

    # You can also observe the predicted probability and the predicted class:
    print("Predicted probabilities:", gpc.predict_proba([x_clf])[0])
    print("Predicted class:", gpc.predict([x_clf])[0])
