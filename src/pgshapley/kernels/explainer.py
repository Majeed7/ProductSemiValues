import math
import time
import numpy as np

"""Explainer backends for product-form kernels.

Provides ProductKernelLocalExplainer and RBFLocalExplainer that compute
per-feature Shapley values using multiple numerical backends (NumPy/JAX).
"""

from product_games_shapley import ProductGamesShapleyNumpy, ProductGamesShapleyJax

try:
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
    '''
    ProductKernelLocalExplainer(model)
    Base class for computing local (per-instance) Shapley-style explanations
    for models with product-form feature kernels (e.g., RBF product kernels).
    Implements utilities to extract training data and dual coefficients from
    various scikit-learn-style estimators, compute feature-wise kernel vectors,
    precompute combinatorial Shapley weights, and evaluate per-feature Shapley
    values via multiple numerical backends.
    Key concepts and conventions
    - The explanation is based on a k-1 factorization of a product kernel:
        K_prod(x, X_train) = \prod_{j=0}^{d-1} k_j(x_j, X_train[:, j]).
        For numerical stability the implementation works with shifted per-feature
        kernels K_j = k_j(x_j, X_train[:, j]) - 1.
    - Shapes:
            X_train : (n, d)      -- training inputs stored by the wrapped model
            alpha   : (n,)        -- dual coefficients or equivalent weight vector
            K       : (d, n)      -- shifted per-feature kernel values (k - 1)
            explain(...) -> (d,)  -- returned per-feature Shapley values
    - The attribute null_game stores the model baseline / intercept when available.
    Parameters
    - model: A fitted scikit-learn compatible estimator using a kernel that
        factorizes across features (examples: SVM/SVR, KernelRidge, GaussianProcessRegressor,
        GaussianProcessClassifier wrappers). The class probes common attributes:
        support_vectors_, X_fit_, X_train_, base_estimator_.X_train_, dual_coef_,
        alpha_, intercept_. If the wrapped estimator does not expose these attributes
        a ValueError is raised when attempting to extract data.
    Public methods
    - get_X_train():
            Return the training inputs (n, d) by inspecting the wrapped model.
            Raises ValueError for unsupported estimator structures.
    - get_alpha():
            Extracts the model's dual coefficients / alpha vector (n,) and sets
            self.null_game when the model exposes an intercept or when an analytic
            baseline can be derived. Raises ValueError for unsupported estimator types.
    - precompute_mu(d: int) -> np.ndarray:
            Static helper returning Shapley combinatorial weights mu[q] =
            q! (d-q-1)! / d! for q in 0..d-1. Returns a 1-D array of length d.
    - compute_kernel_vectors(X, x, gamma) -> list[np.ndarray]:
            Compute per-feature kernel vectors k_j(x_j, X[:, j]) for j=0..d-1.
            Returns a list of d arrays each of shape (n,). By default this calls an
            RBF kernel per feature; subclasses may override for different kernels.
    - explain(x, gamma, method='logspace_jax', m_q=None) -> np.ndarray:
            Compute per-feature Shapley values for a single instance x.
            Parameters:
                x      : (d,) input instance to explain
                gamma  : kernel hyperparameter passed to per-feature kernel computation
                method : one of
                                 {'logspace_numpy', 'logspace_jax',
                                    'prefix_scan_numpy', 'prefix_scan_jax'}
                                 selecting the numerical backend for Gauss–Legendre integration
                                 and stability strategy (log-space vs prefix/suffix scan).
                m_q    : number of Gauss–Legendre nodes. If None, set to ceil(d/2)
                                 (exactness for polynomials up to degree d-1).
            Returns:
                numpy array of shape (d,) containing the per-feature Shapley values.
            Raises:
                ValueError for unknown method or when model interrogation fails.
    Notes and implementation details
    - The class is kernel-agnostic at the API level but the default compute_kernel_vectors
        uses an RBF call per feature; subclasses (e.g., RBFLocalExplainer) can infer
        or manage gamma automatically.
    - The implemention expects the model's prediction to admit a linear expansion
        in duals: f(x) = sum_i alpha_i * prod_j k_j(x_j, X_train[i,j]) + constant.
        The explain routine forms a phi matrix of per-feature contributions
        (d, n) via numerical integration (Gauss–Legendre) and returns
        feature-wise sums (phi * alpha). Different backends trade numerical stability
        and performance (NumPy vs JAX, log-domain vs prefix-scan).
    - This class does not modify the wrapped model; it only reads required attributes.
    - For large n or d choose the backend and m_q to balance accuracy and performance.
    Example (conceptual)
            explainer = ProductKernelLocalExplainer(fitted_model)
            shap_vals = explainer.explain(x_instance, gamma=0.5, method='logspace_numpy')
    '''


    def __init__(self, model):
        """Create an explainer for a fitted kernel model.

        Args:
            model: fitted scikit-learn estimator exposing training data and
                dual coefficients (support_vectors_, X_fit_, X_train_, alpha_, dual_coef_, etc.).
        """
        self.model = model
        self.X_train = self.get_X_train()
        self.alpha = self.get_alpha()
        self.n, self.d = self.X_train.shape
        self.null_game = 0.0  # set in get_alpha() when applicable

    def get_X_train(self):
        """Return training inputs (n, d) from the wrapped model.

        Probes common scikit-learn attributes and raises ValueError if none match.
        """
        if hasattr(self.model, "support_vectors_"):
            return self.model.support_vectors_
        if hasattr(self.model, "X_fit_"):
            return self.model.X_fit_
        if hasattr(self.model, "X_train_"):
            return self.model.X_train_
        if hasattr(self.model, "base_estimator_") and hasattr(self.model.base_estimator_, "X_train_"):
            return self.model.base_estimator_.X_train_
        raise ValueError("Unsupported model type for Shapley value computation (X_train).")

    def get_alpha(self):
        """
        Extract dual coefficients (alpha) and set `null_game` when available.

        Returns an (n,) NumPy array.
        """
        if hasattr(self.model, "dual_coef_"):
            self.null_game = float(np.asarray(self.model.intercept_).ravel()[0])
            return np.asarray(self.model.dual_coef_, dtype=np.float64).ravel()

        if hasattr(self.model, "alpha_"):
            alpha = np.asarray(self.model.alpha_, dtype=np.float64).ravel()
            self.null_game = float(np.sum(alpha))
            return alpha

        if hasattr(self.model, "dual_coef_") and hasattr(self.model, "intercept_"):
            self.null_game = float(np.asarray(self.model.intercept_).ravel()[0])
            return np.asarray(self.model.dual_coef_, dtype=np.float64).ravel()

        raise ValueError("Unsupported model type for Shapley value computation (alpha).")

    @staticmethod
    def precompute_mu(d: int) -> np.ndarray:
        """
        Shapley weights mu[q] = q!(d-q-1)! / d!
        """
        unnormalized = [(math.factorial(q) * math.factorial(d - q - 1)) for q in range(d)]
        return np.array(unnormalized) / math.factorial(d)

    def compute_kernel_vectors(self, X, x, gamma):
        """
        Returns list of length d; each element is (n,) vector k_j(x_j, X_train[:,j]).
        """
        kvs = []
        for j in range(self.d):
            kv = rbf_kernel(X[:, j].reshape(-1, 1), np.array([[x[j]]]), gamma=gamma).squeeze()
            kvs.append(kv)
        return kvs

    def explain(self, x, gamma, method: str = 'logspace_jax', m_q: int | None = None):
        """
        Compute per-feature Shapley values for one instance x using selected backend.

        method ∈ {'logspace_numpy', 'logspace_jax', 'prefix_scan_numpy', 'prefix_scan_jax'}
        m_q   : Gauss–Legendre nodes (if None, uses ceil(d/2) for exactness on degree d-1)
        """
        kernel_vectors = self.compute_kernel_vectors(self.X_train, x, gamma)
        # kernel_vectors is a list of d arrays each of shape (n,)
        # Stack to (n, d) for the (m, d) convention
        K = np.stack(kernel_vectors, axis=1).astype(np.float64) - 1.0  # (n, d)
        alpha = np.asarray(self.alpha, dtype=np.float64)
        d = self.d

        if m_q is None:
            m_q = (d + 1) // 2  # exactness for degree (d-1)

        ## log space Gauss–Legendre backends
        elif method == 'logspace_numpy':
            phi = ProductGamesShapleyNumpy().phi_matrix_logspace(K, m_q)  # (n, d)
            out = (phi * alpha[:, None]).sum(axis=0)  # (d,)

            return out

        elif method == 'logspace_jax':
            phi = ProductGamesShapleyJax().phi_matrix_logspace(K, m_q)  # (n, d)
            out = (phi * alpha[:, None]).sum(axis=0)  # (d,)

            return out

        ## prefix /suffix Gauss Legendre backends
        elif method == 'prefix_scan_numpy':
            phi = ProductGamesShapleyNumpy().phi_matrix_prefix_scan(K, m_q)  # (n, d)
            out = (phi * alpha[:, None]).sum(axis=0)  # (d,)

            return out

        elif method == 'prefix_scan_jax':
            phi = ProductGamesShapleyJax().phi_matrix_prefix_scan(K, m_q)  # (n, d)
            out = (phi * alpha[:, None]).sum(axis=0)  # (d,)

            return out
        
        else:
            raise ValueError("Unknown method. Choose from "
                             "{'logspace_numpy','logspace_jax','prefix_scan_numpy','prefix_scan_jax'}.")


class RBFLocalExplainer(ProductKernelLocalExplainer):
    """
    Specialization for RBF kernels to obtain gamma automatically.
    """

    def __init__(self, model):
        super().__init__(model)
        self.gamma = self.get_gamma()

    def get_gamma(self):
        """Infer an RBF `gamma` value from common model attributes.

        Returns a scalar gamma (1 / (2 * length_scale^2)) when available,
        or raises ValueError if it cannot be inferred.
        """
        if hasattr(self.model, "_gamma"):
            return float(self.model._gamma)

        if hasattr(self.model, "gamma"):
            g = self.model.gamma
            if g is not None:
                return float(g)
            if getattr(self.model, "kernel", None) == "rbf":
                return 1.0 / self.model.X_fit_.shape[1]

        if hasattr(self.model, "kernel_") and hasattr(self.model.kernel_, "length_scale"):
            ls = self.model.kernel_.length_scale
            ls2 = np.mean(np.asarray(ls, dtype=np.float64) ** 2)
            return float(1.0 / (2.0 * ls2))

        raise ValueError("Cannot infer gamma for the provided model.")

    def explain(self, x, method: str = 'esp-collective', m_q: int | None = None):
        return super().explain(x=np.asarray(x, dtype=np.float64),
                               gamma=self.gamma,
                               method=method,
                               m_q=m_q)

if __name__ == "__main__":
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF
    from sklearn.datasets import make_regression
    from sklearn.model_selection import train_test_split
    from sklearn.kernel_ridge import KernelRidge

    X, y = make_regression(n_samples=500, n_features=100, random_state=40)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

    kernel = RBF(1.0, (1e-3, 1e3))
    gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=0, alpha=1e-2, normalize_y=False)
    print("Fitting GPR...")
    t0 = time.time()
    gpr.fit(X_train, y_train)
    print(f"GPR fit time: {time.time()-t0:.3f}s  |  learned kernel: {gpr.kernel_}")

    explainer = RBFLocalExplainer(gpr)
    x = X_train[0]

    methods = [
        ("logspace_numpy", True),
        ("logspace_jax", True),
        ("prefix_scan_numpy", True),
        ("prefix_scan_jax", True),
    ]

    m_q = 150

    print("the sume should be: ", gpr.alpha_.sum() - gpr.predict([x]))
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

