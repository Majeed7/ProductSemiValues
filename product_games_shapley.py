import numpy as np

# Optional JAX support
try:
    import jax
    import jax.numpy as jnp
    from jax import lax

    JAX_AVAILABLE = True
except Exception:
    JAX_AVAILABLE = False


def _gauss_legendre_01_numpy(m_q: int, dtype=np.float64):
    """
    Gauss–Legendre nodes/weights mapped from [-1,1] to [0,1].
    """
    x, w = np.polynomial.legendre.leggauss(m_q)
    x = 0.5 * (x + 1.0)
    w = 0.5 * w
    return x.astype(dtype, copy=False), w.astype(dtype, copy=False)


class ProductGamesShapleyNumpy:
    """
    Alpha-free product-game Shapley factors in NumPy.

    This class operates on the product-game array K of shape (d, m).
    It returns a matrix Phi of shape (d, m) such that for any coefficients
    alpha of shape (m,), the Shapley values are:

        shapley = (Phi * alpha[None, :]).sum(axis=1)

    Here K is the per-feature factor used inside the product:

        v_S(m) = prod_{j in S} (1 + K[j, m])    (up to your model-specific shift)

    The implementations match the backends in `explainer.py` but do not depend
    on any model or kernel; they only require K and quadrature size m_q.
    """

    def phi_matrix_prefix_scan(self, K: np.ndarray, m_q: int) -> np.ndarray:
        K = np.asarray(K, dtype=np.float64)
        d, m = K.shape

        x, w = _gauss_legendre_01_numpy(m_q, dtype=np.float64)
        X = x[:, None, None]  # (m_q,1,1)
        B = 1.0 + X * K[None, :, :]  # (m_q,d,m)

        pref = np.cumprod(B, axis=1)
        pref = np.concatenate(
            [np.ones((B.shape[0], 1, m), dtype=B.dtype), pref[:, :-1, :]], axis=1
        )

        suf = np.cumprod(B[:, ::-1, :], axis=1)[:, ::-1, :]
        suf = np.concatenate(
            [suf[:, 1:, :], np.ones((B.shape[0], 1, m), dtype=B.dtype)], axis=1
        )

        Q_no_i = pref * suf  # (m_q,d,m)
        acc = (w[:, None, None] * Q_no_i).sum(axis=0)  # (d,m)
        return K * acc

    def phi_matrix_logspace(self, K: np.ndarray, m_q: int, eps: float = 1e-12) -> np.ndarray:
        """Compute Phi (d,m) using log-space shared product.

        This is memory-lean in the shared product, but still returns Phi (d,m).
        """
        K = np.asarray(K, dtype=np.float64)
        d, m = K.shape

        x, w = _gauss_legendre_01_numpy(m_q, dtype=np.float64)

        log_abs_P = np.zeros((m_q, m), dtype=np.float64)
        sign_P = np.ones((m_q, m), dtype=np.float64)

        for j in range(d):
            t = 1.0 + np.outer(x, K[j, :])  # (m_q,m)
            sign_P *= np.sign(t)
            log_abs_P += np.log(np.maximum(np.abs(t), eps), dtype=np.float64)

        Qint = np.empty((d, m), dtype=np.float64)
        wa = w[:, None]
        for i in range(d):
            denom = 1.0 + np.outer(x, K[i, :])  # (m_q,m)
            integrand_sign = sign_P * np.sign(denom)
            integrand_log = log_abs_P - np.log(np.maximum(np.abs(denom), eps), dtype=np.float64)
            Qint[i, :] = (wa * (integrand_sign * np.exp(integrand_log))).sum(axis=0)

        return K * Qint


class ProductGamesShapleyJax:
    """
    Product-game Shapley factors in JAX.

    Returns Phi (d,m) in NumPy format.
    """

    def __init__(self):
        if not JAX_AVAILABLE:
            raise RuntimeError("JAX is not available on this system.")

    @staticmethod
    @jax.jit
    def _phi_prefix_core(K, x, w):
        B = 1.0 + x[:, None, None] * K[None, :, :]  # (m_q,d,m)
        pref = lax.cumprod(B, axis=1)
        pref = jnp.concatenate(
            [jnp.ones((B.shape[0], 1, B.shape[2]), dtype=B.dtype), pref[:, :-1, :]], axis=1
        )
        suf = lax.cumprod(B[:, ::-1, :], axis=1)[:, ::-1, :]
        suf = jnp.concatenate(
            [suf[:, 1:, :], jnp.ones((B.shape[0], 1, B.shape[2]), dtype=B.dtype)], axis=1
        )
        Q = pref * suf
        acc = (w[:, None, None] * Q).sum(axis=0)  # (d,m)
        return K * acc

    def phi_matrix_prefix_scan(self, K: np.ndarray, m_q: int) -> np.ndarray:
        K = np.asarray(K)

        x_np, w_np = np.polynomial.legendre.leggauss(m_q)
        x_np = 0.5 * (x_np + 1.0)
        w_np = 0.5 * w_np

        # dtype policy: prefer float32 on accelerators
        dtype = np.result_type(K.dtype, np.float32)
        try:
            platforms = {dev.platform for dev in jax.devices()}
        except Exception:
            platforms = set()
        if (jax.default_backend() != "cpu") or (platforms & {"gpu", "tpu", "metal"}):
            dtype = np.float32
            K = K.astype(np.float32, copy=False)
        x = jnp.asarray(x_np, dtype=dtype)
        w = jnp.asarray(w_np, dtype=dtype)
        Kj = jnp.asarray(K, dtype=dtype)

        out = self._phi_prefix_core(Kj, x, w)
        return np.asarray(out)

    @staticmethod
    @jax.jit
    def _phi_logspace_core(K, x, w, eps):
        x = x[:, None]  # (m_q,1)
        w = w[:, None]  # (m_q,1)
        d, m = K.shape

        def scan_step(carry, k_j):
            log_abs_P, sign_P = carry
            t = 1.0 + x * k_j[None, :]
            sign_P = sign_P * jnp.sign(t)
            log_abs_P = log_abs_P + jnp.log(jnp.maximum(jnp.abs(t), eps))
            return (log_abs_P, sign_P), None

        init = (
            jnp.zeros((x.shape[0], m), K.dtype),
            jnp.ones((x.shape[0], m), K.dtype),
        )
        (log_abs_P, sign_P), _ = lax.scan(scan_step, init, K)

        def per_feature(k_i):
            denom = 1.0 + x * k_i[None, :]
            integrand_sign = sign_P * jnp.sign(denom)
            integrand_log = log_abs_P - jnp.log(jnp.maximum(jnp.abs(denom), eps))
            Qint = jnp.sum(w * (integrand_sign * jnp.exp(integrand_log)), axis=0)  # (m,)
            return k_i * Qint  # (m,)

        return jax.vmap(per_feature, in_axes=0)(K)  # (d,m)

    def phi_matrix_logspace(self, K: np.ndarray, m_q: int, eps: float = 1e-100) -> np.ndarray:
        K = np.asarray(K)

        x_np, w_np = np.polynomial.legendre.leggauss(m_q)
        x_np = 0.5 * (x_np + 1.0)
        w_np = 0.5 * w_np

        # dtype policy: prefer float32 on accelerators
        dtype = np.result_type(K.dtype, np.float32)
        try:
            platforms = {dev.platform for dev in jax.devices()}
        except Exception:
            platforms = set()
        if (jax.default_backend() != "cpu") or (platforms & {"gpu", "tpu", "metal"}):
            dtype = np.float32
            K = K.astype(np.float32, copy=False)
        x = jnp.asarray(x_np, dtype=dtype)
        w = jnp.asarray(w_np, dtype=dtype)
        Kj = jnp.asarray(K, dtype=dtype)

        out = self._phi_logspace_core(Kj, x, w, eps)
        return np.asarray(out)
