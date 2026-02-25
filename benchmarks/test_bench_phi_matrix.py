import pytest
import numpy as np

import jax
jax.config.update("jax_enable_x64", True)

from pgshapley.product_games.shapley import ProductGamesShapleyNumpy, ProductGamesShapleyJax, JAX_AVAILABLE


def _phi_matrix_methods():
    np_obj = ProductGamesShapleyNumpy()
    yield pytest.param(np_obj.phi_matrix_prefix_scan, id="numpy_prefix_scan")
    yield pytest.param(np_obj.phi_matrix_logspace, id="numpy_logspace")
    if JAX_AVAILABLE:
        jax_obj = ProductGamesShapleyJax()
        yield pytest.param(jax_obj.phi_matrix_prefix_scan, id="jax_prefix_scan")
        yield pytest.param(jax_obj.phi_matrix_logspace, id="jax_logspace")


@pytest.fixture(params=list(_phi_matrix_methods()))
def phi_fn(request):
    return request.param


def _make_K(d, seed=0):
    rng = np.random.default_rng(seed)
    c = 5.0 / np.sqrt(d)
    a = np.exp(rng.uniform(-c, c, size=d))
    return (a - 1.0).reshape(d, 1)


@pytest.mark.parametrize("d", [10, 50, 100, 500, 1000])
def test_bench_exact_quadrature(benchmark, phi_fn, d):
    K = _make_K(d)
    m_q = max(1, d // 2)
    # Warmup.
    phi_fn(K, m_q)
    benchmark(phi_fn, K, m_q)


@pytest.mark.parametrize("d", [100, 1000, 10000, 100000])
def test_bench_fixed_quadrature(benchmark, phi_fn, d):
    K = _make_K(d)
    m_q = 100
    # Warmup.
    phi_fn(K, m_q)
    benchmark(phi_fn, K, m_q)
