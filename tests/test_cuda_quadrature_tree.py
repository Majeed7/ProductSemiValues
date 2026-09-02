import numpy as np
import pytest


def _has_cuda_cupy():
    try:
        import cupy as cp

        return cp.cuda.runtime.getDeviceCount() > 0
    except Exception:
        return False


@pytest.mark.skipif(not _has_cuda_cupy(), reason="CUDA CuPy is unavailable")
@pytest.mark.parametrize("classifier", [False, True])
def test_cuda_quadrature_tree_matches_cpu(classifier):
    from quadrashap import TreeExplainer
    from sklearn.datasets import make_classification, make_regression
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

    if classifier:
        X, y = make_classification(
            n_samples=300, n_features=12, n_informative=8, random_state=7
        )
        X = np.column_stack((X, np.zeros((len(X), 2))))
        model = RandomForestClassifier(
            n_estimators=7, max_depth=7, random_state=9
        ).fit(X, y)
    else:
        X, y = make_regression(
            n_samples=300, n_features=12, n_informative=8, random_state=7
        )
        X = np.column_stack((X, np.zeros((len(X), 2))))
        model = RandomForestRegressor(
            n_estimators=7, max_depth=7, random_state=9
        ).fit(X, y)

    # The scalar case crosses the large-batch treelet dispatch; the
    # multi-output classifier exercises the full-state fallback.
    n_test = 256 if not classifier else 65
    X_test = np.ascontiguousarray(X[:n_test])
    cpu = TreeExplainer(
        model, tree_solver="quadrature_tree", device="cpu"
    ).shap_values(X_test, check_additivity=False)
    gpu_explainer = TreeExplainer(
        model, tree_solver="quadrature_tree", device="cuda"
    )
    gpu = gpu_explainer.shap_values(X_test, check_additivity=True)
    np.testing.assert_allclose(gpu, cpu, rtol=2e-10, atol=2e-10)
    if not classifier:
        treelet_gpu = (
            gpu_explainer._backend._cuda_solver._explain_scalar_treelets(
                X_test
            )[:, :, 0]
        )
        np.testing.assert_allclose(
            treelet_gpu, cpu, rtol=2e-10, atol=2e-10
        )
