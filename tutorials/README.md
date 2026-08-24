# QuadraSHAP tutorials

These notebooks introduce the two main QuadraSHAP use cases with small,
fully reproducible examples:

| Notebook | What it demonstrates |
|---|---|
| [`tree_models.ipynb`](tree_models.ipynb) | Exact path-dependent TreeSHAP for a scikit-learn decision tree, checked against exhaustive coalition enumeration |
| [`kernel_methods.ipynb`](kernel_methods.ipynb) | Exact local Shapley values for an RBF `KernelRidge` model, checked against exhaustive product-game enumeration |

Both notebooks:

- train a small model on synthetic data;
- explain one prediction with QuadraSHAP;
- implement the corresponding naive Shapley calculation from scratch;
- verify attribution agreement and the Shapley additivity identity; and
- time both methods and print the speedup measured on the current machine.

The explanations follow the derivation and notation in:

> Mohammadi, M., Reznikov, G., Sinitcyn, P., Muandet, K., and Chau, S. L.
> **QuadraSHAP: Stable and Scalable Shapley Values for Product Games via
> Gauss-Legendre Quadrature.** arXiv:2605.05870v2, 2026.

In particular, the notebooks unpack the Beta-integral reduction (Proposition
2), Gauss-Legendre exactness and approximation result (Proposition 3), shared
log-space and scan computation (Proposition 4), the product-kernel value
function (Section 4.1), and the optimized tree traversal (Section 4.2 and
Appendix C).

The naive methods enumerate all `2**d` feature coalitions. They are included
only as transparent correctness and timing baselines, so increase the feature
counts cautiously.

## Running the notebooks

From the repository root, install QuadraSHAP and the tutorial dependencies:

```bash
python -m pip install -e .
python -m pip install jupyter scikit-learn
```

Then start Jupyter:

```bash
jupyter lab tutorials/
```

The tree notebook automatically uses the native C++ quadrature backend when
it is installed. Otherwise it uses the pure-Python implementation. The kernel
notebook uses the NumPy log-space backend, so neither notebook requires a GPU.
