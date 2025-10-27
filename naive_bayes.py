import numpy as np
from dataclasses import dataclass
from typing import Optional
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB, ComplementNB

@dataclass
class NBProducts:
    classes_: np.ndarray                # (K,)
    prior: np.ndarray                   # (K,)   p(c)
    factors: np.ndarray                 # (n, K, d)  p(x_i | c) per (sample, class, feature)
    # convenience: per-sample unnormalized posteriors (all multiplicative)
    joint_products: np.ndarray          # (n, K)  p(c) * ∏_i p(x_i|c)

def _binarize_if_needed(model, X: np.ndarray) -> np.ndarray:
    if hasattr(model, "binarize") and model.binarize is not None:
        return (X > model.binarize).astype(np.int8)
    return X

def product_factors_naive_bayes(model, X, *, eps: float = 1e-300) -> NBProducts:
    """
    Return multiplicative per-feature factors for a fitted sklearn NB model.
    
    Parameters
    ----------
    model : GaussianNB | MultinomialNB | BernoulliNB | ComplementNB (fitted)
    X : array-like of shape (n, d)  — must be preprocessed exactly like training.
    eps : tiny floor to avoid zeros in probabilities
    
    Returns
    -------
    NBProducts with:
      - classes_: model.classes_
      - prior: p(c)  (uniform if sklearn kept None)
      - factors: p(x_i | c)  shape (n, K, d)
      - joint_products: p(c) * ∏_i p(x_i | c)  shape (n, K)
    """
    X = np.asarray(X)
    n, d = X.shape
    classes = np.asarray(model.classes_)
    K = classes.size

    # Priors in probability space
    if getattr(model, "class_log_prior_", None) is not None:
        prior = np.exp(np.asarray(model.class_log_prior_))
    else:
        prior = np.ones(K) / K  # sklearn sometimes leaves None

    factors = np.empty((n, K, d), dtype=float)

    if isinstance(model, (MultinomialNB, ComplementNB)):
        # WARNING (ComplementNB): these are complement-weight-based “pseudo-probabilities”.
        # They reproduce sklearn scores multiplicatively but are not literal p(x_i|c) from a generative model.
        # feature_log_prob_: (K, d)
        phi = np.exp(np.asarray(model.feature_log_prob_))   # φ_{c,i} ≈ p(feature i | c)
        # For counts TF/TF-IDF X (as trained): factor = φ_{c,i} ** x_i
        # Broadcast (n,d) with (K,d) -> (n,K,d)
        for c in range(K):
            factors[:, c, :] = np.power(phi[c, :], X)

    elif isinstance(model, BernoulliNB):
        Xb = _binarize_if_needed(model, X)
        p1 = np.exp(np.asarray(model.feature_log_prob_))       # (K, d)  p(x_i=1|c)
        p1 = np.clip(p1, eps, 1 - 1e-15)
        p0 = 1.0 - p1
        # factors = p1 if x_i=1 else p0
        for c in range(K):
            factors[:, c, :] = Xb * p1[c, :] + (1 - Xb) * p0[c, :]

    elif isinstance(model, GaussianNB):
        mu = np.asarray(model.theta_)   # (K, d)
        var = np.asarray(model.var_)    # (K, d) (already smoothed inside sklearn)
        var = np.clip(var, eps, np.inf)
        inv_var = 1.0 / var
        norm = 1.0 / np.sqrt(2.0 * np.pi * var)               # (K, d)
        # p(x_i | c) = norm * exp(-0.5 * (x - mu)^2 / var)
        # vectorized per class
        for c in range(K):
            dif2 = (X - mu[c, :]) ** 2                         # (n, d)
            factors[:, c, :] = norm[c, :] * np.exp(-0.5 * dif2 * inv_var[c, :])

    else:
        raise TypeError(f"Unsupported NB model type: {type(model)}")

    # tiny floor against exact zeros, without upsetting relative magnitudes
    np.maximum(factors, eps, out=factors)

    # joint unnormalized posterior per sample/class in product form
    # p(c) * ∏_i p(x_i|c)
    joint_products = prior[None, :] * np.prod(factors, axis=2)

    return NBProducts(
        classes_=classes,
        prior=prior,
        factors=factors,
        joint_products=joint_products,
    )


if __name__ == "__main__":
    # simple test
    from sklearn.datasets import load_iris
    from sklearn.naive_bayes import GaussianNB

    X, y = load_iris(return_X_y=True)
    gnb = GaussianNB().fit(X, y)

    prods = product_factors_naive_bayes(gnb, X[:5])
    print("classes:", prods.classes_)
    print("prior:", prods.prior)                 # p(c)
    print("factors shape:", prods.factors.shape) # (3, K, d)
    print("joint_products shape:", prods.joint_products.shape)

    # Example: multiplicative factors for sample 0, class c
    c = 1
    print("per-feature factors (sample 0, class 1):")
    print(np.round(prods.factors[0, c], 6))      # all in probability space
    print("unnormalized posterior product for that sample/class:",
        prods.joint_products[0, c])
