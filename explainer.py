import math
import time
import numpy as np

# Optional (only needed for the JAX method)
try:
    # import os
    # os.environ["JAX_PLATFORM_NAME"] = "cpu"
    import jax
    import jax.numpy as jnp
    from jax import lax
    JAX_AVAILABLE = True
    print("Backend forced to:", jax.default_backend())
    print("Devices:", jax.devices())
except Exception:
    JAX_AVAILABLE = False

from sklearn.metrics.pairwise import rbf_kernel


class ProductKernelLocalExplainer:
    """
    Base class for product-kernel local explanations (k-1 factorization).
    Supports multiple computation backends for the Shapley integral identity.
    """

    def __init__(self, model):
        """
        Args:
            model: scikit-learn model (SVM/SVR/KRR/GP) with an RBF-like product kernel
        """
        self.model = model
        self.X_train = self.get_X_train()
        self.alpha = self.get_alpha()
        self.n, self.d = self.X_train.shape
        self.null_game = 0.0  # set in get_alpha() when applicable

    # -------------------- model plumbing --------------------

    def get_X_train(self):
        if hasattr(self.model, "support_vectors_"):      # SVM/SVR
            return self.model.support_vectors_
        if hasattr(self.model, "X_fit_"):                # KernelRidge
            return self.model.X_fit_
        if hasattr(self.model, "X_train_"):              # GPR (sklearn<=1.5)
            return self.model.X_train_
        if hasattr(self.model, "base_estimator_") and hasattr(self.model.base_estimator_, "X_train_"):  # GPC
            return self.model.base_estimator_.X_train_
        raise ValueError("Unsupported model type for Shapley value computation (X_train).")

    def get_alpha(self):
        if hasattr(self.model, "dual_coef_"):  # SVM/SVR
            self.null_game = float(np.asarray(self.model.intercept_).ravel()[0])
            return np.asarray(self.model.dual_coef_, dtype=np.float64).ravel()

        if hasattr(self.model, "alpha_"):      # GPR: alpha_ = K^{-1} y (size n)
            alpha = np.asarray(self.model.alpha_, dtype=np.float64).ravel()
            self.null_game = float(np.sum(alpha))
            return alpha

        if hasattr(self.model, "dual_coef_") and hasattr(self.model, "intercept_"):
            # fallback for other dual models
            self.null_game = float(np.asarray(self.model.intercept_).ravel()[0])
            return np.asarray(self.model.dual_coef_, dtype=np.float64).ravel()

        raise ValueError("Unsupported model type for Shapley value computation (alpha).")

    # -------------------- combinatorial coefficients --------------------

    @staticmethod
    def precompute_mu(d: int) -> np.ndarray:
        """
        Shapley weights mu[q] = q!(d-q-1)! / d!
        """
        unnormalized = [(math.factorial(q) * math.factorial(d - q - 1)) for q in range(d)]
        return np.array(unnormalized) / math.factorial(d)

    # -------------------- ESP (dynamic programming) --------------------

    @staticmethod
    def compute_elementary_symmetric_polynomials(kernel_vectors, scale: bool = False):
        """
        DP for ESPs e_q over feature-wise vectors k_j (each shape (m,)).
        Returns list [e_0,...,e_d], each shape (m,).
        """
        max_abs_k = 1.0
        if scale:
            max_abs_k = max(np.max(np.abs(k)) for k in kernel_vectors) or 1.0
        scaled_kernel = [k / max_abs_k for k in kernel_vectors]

        e = [np.ones_like(scaled_kernel[0], dtype=np.float64)]
        for k in scaled_kernel:
            new_e = [np.zeros_like(e[0])] * (len(e) + 1)
            new_e[0] = -k * e[0]
            for i in range(1, len(e)):
                new_e[i] = e[i - 1] - k * e[i]
            new_e[len(e)] = e[-1].copy()
            e = new_e

        n = len(scaled_kernel)
        elementary = [np.ones_like(e[0])]
        for r in range(1, n + 1):
            sign = (-1) ** r
            elementary.append(sign * e[n - r] * (max_abs_k ** r))
        return elementary

    # -------------------- Gauss–Legendre backends --------------------

    @staticmethod
    def logspace_numpy(K: np.ndarray, alpha: np.ndarray, m_q: int):
        """
        Memory-lean, log-space shared product.
        K: (d,m), alpha: (m,), returns (d,)
        """
        K = np.asarray(K, dtype=np.float64)
        alpha = np.asarray(alpha, dtype=np.float64)
        d, m = K.shape

        # Nodes/weights on [0,1]
        x, w = np.polynomial.legendre.leggauss(m_q)
        x = 0.5 * (x + 1.0)
        w = 0.5 * w

        # Shared product per node&column in log-space
        log_abs_P = np.zeros((m_q, m), dtype=np.float64)
        sign_P = np.ones((m_q, m), dtype=np.float64)

        eps = 1e-12
        for j in range(d):
            t = 1.0 + np.outer(x, K[j, :])  # (m_q, m)
            sign_P *= np.sign(t)
            log_abs_P += np.log(np.maximum(np.abs(t), eps), dtype=np.float64)

        result = np.zeros(d, dtype=np.float64)
        wa = w[:, None]                      # (m_q,1)
        a = alpha[None, :]                   # (1,m)
        for i in range(d):
            denom = 1.0 + np.outer(x, K[i, :])  # (m_q, m)
            integrand_sign = sign_P * np.sign(denom)
            integrand_log = log_abs_P - np.log(np.maximum(np.abs(denom), eps), dtype=np.float64)
            Qint = (wa * (integrand_sign * np.exp(integrand_log))).sum(axis=0)  # (m,)
            result[i] = (K[i, :] * a * Qint).sum()
        return result

    # ---- JAX wrapper for log space method ----
    @staticmethod
    @jax.jit
    def _gauss_shared_jax_core(K, alpha, x, w, eps):
        """
        JIT-safe pure core: all args are arrays or scalars, no `self`.
        K: (d,m)  alpha: (m,)  x:(m_q,), w:(m_q,)
        returns: (d,)
        """
        # reshape nodes/weights for broadcasting
        x = x[:, None]      # (m_q,1)
        w = w[:, None]      # (m_q,1)

        d, m = K.shape
        # 1) shared product over features in log-space via scan across feature axis
        def scan_step(carry, k_j):
            log_abs_P, sign_P = carry                # both (m_q,m)
            t = 1.0 + x * k_j[None, :]              # (m_q,m)
            sign_P = sign_P * jnp.sign(t)
            log_abs_P = log_abs_P + jnp.log(jnp.maximum(jnp.abs(t), eps))
            return (log_abs_P, sign_P), None

        init = (jnp.zeros((x.shape[0], m), K.dtype),
                jnp.ones((x.shape[0], m), K.dtype))
        (log_abs_P, sign_P), _ = lax.scan(scan_step, init, K)

        # 2) per-feature division & integration, vectorized with vmap
        def per_feature(k_i):
            denom = 1.0 + x * k_i[None, :]            # (m_q,m)
            integrand_sign = sign_P * jnp.sign(denom)
            integrand_log  = log_abs_P - jnp.log(jnp.maximum(jnp.abs(denom), eps))
            Qint = jnp.sum(w * (integrand_sign * jnp.exp(integrand_log)), axis=0)  # (m,)
            return jnp.sum(k_i * alpha * Qint)        # scalar

        return jax.vmap(per_feature, in_axes=0)(K)     # (d,)

    def logspace_jax(self, K, alpha, m_q: int, eps: float = 1e-100):
        """
        Non-jitted wrapper: builds GL nodes/weights and calls the jitted core.
        K: (d,m) numpy/jax array, alpha: (m,)
        """
        # Choose dtype: prefer f32 on accelerators
        dtype = jnp.result_type(getattr(K, 'dtype', jnp.float32),
                                getattr(alpha, 'dtype', jnp.float32),
                                jnp.float32)
        # Build Gauss–Legendre nodes/weights on [0,1] in NumPy, then to JAX
        x_np, w_np = np.polynomial.legendre.leggauss(m_q)
        x_np = 0.5 * (x_np + 1.0)      # map to [0,1]
        w_np = 0.5 * w_np

        x = jnp.asarray(x_np, dtype=dtype)
        w = jnp.asarray(w_np, dtype=dtype)
        Kj = jnp.asarray(K, dtype=dtype)
        aj = jnp.asarray(alpha, dtype=dtype)

        out = self._gauss_shared_jax_core(Kj, aj, x, w, eps)
        return np.asarray(out)

    # ------ Prefix/suffix Gauss–Legendre backend ------
    @staticmethod
    def prefix_scan_numpy(K: np.ndarray, alpha: np.ndarray, m_q: int):
        """
        Fully vectorized prefix/suffix per quadrature node (NumPy).
        K: (d,m), alpha: (m,), returns (d,)
        """
        K = np.asarray(K, dtype=np.float64)
        alpha = np.asarray(alpha, dtype=np.float64)
        d, m = K.shape

        x, w = np.polynomial.legendre.leggauss(m_q)
        x = 0.5 * (x + 1.0)
        w = 0.5 * w

        X = x[:, None, None]                 # (m_q,1,1)
        B = 1.0 + X * K[None, :, :]          # (m_q, d, m)

        pref = np.cumprod(B, axis=1)
        pref = np.concatenate([np.ones((B.shape[0], 1, m), B.dtype), pref[:, :-1, :]], axis=1)

        suf = np.cumprod(B[:, ::-1, :], axis=1)[:, ::-1, :]
        suf = np.concatenate([suf[:, 1:, :], np.ones((B.shape[0], 1, m), B.dtype)], axis=1)

        Q_no_i = pref * suf                  # (m_q, d, m)
        acc_vec = (w[:, None, None] * Q_no_i).sum(axis=0)  # (d, m)
        return (K * acc_vec * alpha[None, :]).sum(axis=1)

    # ---- JAX version of Prefix/suffix ----
    @staticmethod
    def _jax_core(K, alpha, x, w):
        B = 1.0 + x[:, None, None] * K[None, :, :]  # (m_q,d,m)
        pref = lax.cumprod(B, axis=1)
        pref = jnp.concatenate([jnp.ones((B.shape[0], 1, B.shape[2]), dtype=B.dtype), pref[:, :-1, :]], axis=1)
        suf = lax.cumprod(B[:, ::-1, :], axis=1)[:, ::-1, :]
        suf = jnp.concatenate([suf[:, 1:, :], jnp.ones((B.shape[0], 1, B.shape[2]), dtype=B.dtype)], axis=1)
        Q = pref * suf
        acc = (w[:, None, None] * Q).sum(axis=0)    # (d,m)
        return (K * acc * alpha[None, :]).sum(axis=1)  # (d,)

    @staticmethod
    def prefix_scan_jax(K: np.ndarray, alpha: np.ndarray, m_q: int):
        """
        JAX/XLA prefix/suffix backend.
        """
        if not JAX_AVAILABLE:
            raise RuntimeError("JAX is not available on this system.")
        # Nodes/weights on [0,1]
        x_np, w_np = np.polynomial.legendre.leggauss(m_q)
        x_np = 0.5 * (x_np + 1.0)
        w_np = 0.5 * w_np

        # dtype policy: prefer float32 on accelerators
        dtype = np.result_type(K.dtype, alpha.dtype, np.float32)
        try:
            platforms = {dev.platform for dev in jax.devices()}
        except Exception:
            platforms = set()
        if (jax.default_backend() != "cpu") or (platforms & {"gpu", "tpu", "metal"}):
            dtype = np.float32
            K = K.astype(np.float32, copy=False)
            alpha = alpha.astype(np.float32, copy=False)
        x = jnp.asarray(x_np, dtype=dtype)
        w = jnp.asarray(w_np, dtype=dtype)
        Kj = jnp.asarray(K, dtype=dtype)
        aj = jnp.asarray(alpha, dtype=dtype)

        _jit_core = jax.jit(ProductKernelLocalExplainer._jax_core)
        out = _jit_core(Kj, aj, x, w)
        return np.asarray(out)

    # -------------------- high-level API --------------------

    def compute_kernel_vectors(self, X, x, gamma):
        """
        Returns list of length d; each element is (n,) vector k_j(x_j, X_train[:,j]).
        """
        kvs = []
        for j in range(self.d):
            kv = rbf_kernel(X[:, j].reshape(-1, 1), np.array([[x[j]]]), gamma=gamma).squeeze()
            kvs.append(kv)
        return kvs

    def explain(self, x, gamma, method: str = 'esp-collective', m_q: int | None = None):
        """
        Compute per-feature Shapley values for one instance x using selected backend.

        method ∈ {'esp-collective', 'gauss-shared', 'gauss-prefix', 'gauss-jax'}
        m_q   : Gauss–Legendre nodes (if None, uses ceil(d/2) for exactness on degree d-1)
        """
        # 1) feature-wise kernel vectors and shifted K = k-1
        kernel_vectors = self.compute_kernel_vectors(self.X_train, x, gamma)  # list of length d, each (n,)
        K = np.asarray(kernel_vectors, dtype=np.float64) - 1.0               # (d, n)
        alpha = np.asarray(self.alpha, dtype=np.float64)
        d = self.d

        if m_q is None:
            m_q = (d + 1) // 2  # exactness for degree (d-1)

        if method == 'esp-collective':
            # Global ESPs E[0..d] (each (n,))
            E = np.zeros((d + 1, K.shape[1]), dtype=np.float64)
            E[0] = 1.0
            # Note: here K means (k_j - 1). For ESPs we need k_j (not K),
            # so use (K+1).
            Kp1 = K + 1.0
            for j in range(d):
                for q in range(j + 1, 0, -1):
                    E[q] += Kp1[j] * E[q - 1]
            mu = self.precompute_mu(d)

            # For each i: synthetic division recurrence
            shap = np.zeros(d, dtype=np.float64)
            for i in range(d):
                k_i = Kp1[i]
                e_prev = np.ones_like(k_i)       # e^{(-i)}_0
                total = mu[0] * e_prev
                for q in range(1, d):
                    e_curr = E[q] - k_i * e_prev
                    total += mu[q] * e_curr
                    e_prev = e_curr
                shap[i] = (alpha * (Kp1[i] - 1.0) * total).sum()  # (k_i-1) * sum_q mu[q] e^{(-i)}_q
            return shap

        ## log space Gauss–Legendre backends
        elif method == 'logspace_numpy':
            return self.logspace_numpy(K, alpha, m_q)
        
        elif method == 'logspace_jax':
            return self.logspace_jax(K, alpha, m_q)

        ## prefix /suffix Gauss Legendre backends
        elif method == 'prefix_scan_numpy':
            return self.prefix_scan_numpy(K, alpha, m_q)

        elif method == 'prefix_scan_jax':
            return self.prefix_scan_jax(K, alpha, m_q)
        
        else:
            raise ValueError("Unknown method. Choose from "
                             "{'esp-collective','gauss-shared','gauss-prefix','gauss-jax'}.")


class RBFLocalExplainer(ProductKernelLocalExplainer):
    """
    Specialization for RBF kernels to obtain gamma automatically.
    """

    def __init__(self, model):
        super().__init__(model)
        self.gamma = self.get_gamma()

    def get_gamma(self):
        if hasattr(self.model, "_gamma"):            # SVM/SVR
            return float(self.model._gamma)

        if hasattr(self.model, "gamma"):             # KernelRidge
            g = self.model.gamma
            if g is not None:
                return float(g)
            # default 'scale' fallback: 1 / n_features
            if getattr(self.model, "kernel", None) == "rbf":
                return 1.0 / self.model.X_fit_.shape[1]

        # GPR with RBF kernel: kernel_.length_scale may be scalar or array
        if hasattr(self.model, "kernel_") and hasattr(self.model.kernel_, "length_scale"):
            ls = self.model.kernel_.length_scale
            # length_scale can be scalar or vector; we interpret gamma = 1 / (2 * ls^2)
            ls2 = np.mean(np.asarray(ls, dtype=np.float64) ** 2)
            return float(1.0 / (2.0 * ls2))

        raise ValueError("Cannot infer gamma for the provided model.")

    def explain(self, x, method: str = 'esp-collective', m_q: int | None = None):
        return super().explain(x=np.asarray(x, dtype=np.float64),
                               gamma=self.gamma,
                               method=method,
                               m_q=m_q)


# -------------------- Benchmark script --------------------
if __name__ == "__main__":
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF
    from sklearn.datasets import make_regression
    from sklearn.model_selection import train_test_split
    from sklearn.kernel_ridge import KernelRidge

    # --- DATA ---
    # WARNING: A GPR scales ~O(n^3). Large n or many restarts can be slow.
    # The user example uses 10,000 features. That is fine for per-feature kernel-vector work,
    # but model fitting time/memory for GPR may dominate. Adjust as needed.
    X, y = make_regression(n_samples=1000, n_features=500, random_state=40)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

    # --- MODEL: Gaussian Process Regressor ---
    kernel = RBF(1.0, (1e-3, 1e3))
    gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=0, alpha=1e-2, normalize_y=False)
    print("Fitting GPR...")
    t0 = time.time()
    gpr.fit(X_train, y_train)
    print(f"GPR fit time: {time.time()-t0:.3f}s  |  learned kernel: {gpr.kernel_}")

    # --- Explainer ---
    explainer = RBFLocalExplainer(gpr)
    x = X_test[0]

    methods = [
        ("esp-collective", True),
        ("logspace_numpy", True),
        ("logspace_jax", True),
        ("prefix_scan_numpy", True),
        ("prefix_scan_jax", JAX_AVAILABLE),
    ]

    # choose m_q (Gauss nodes); for exactness on degree (d-1), m_q=ceil(d/2) is huge for d=10k.
    # Use a practical cap (e.g., 32 or 64). You can tune this.
    m_q = 100

    results = {}
    print("\nBenchmarking methods on a single instance...")
    for name, enabled in methods:
        if not enabled:
            print(f"  - {name:14s} : (skipped; JAX not available)")
            continue
        t0 = time.time()
        vals = explainer.explain(x, method=name, m_q=m_q)
        dt = time.time() - t0
        results[name] = (vals, dt)
        print(f"  - {name:14s} : time = {dt:.3f}s | sum(phi)={np.sum(vals):.6g}")

    print("done.")
    # --- Compare with Kernel Ridge (optional) ---
