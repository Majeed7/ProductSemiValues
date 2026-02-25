import pytest
import numpy as np

from pgshapley.product_games.shapley import ProductGamesShapleyNumpy, ProductGamesShapleyJax, JAX_AVAILABLE

if JAX_AVAILABLE:
    import jax

    jax.config.update("jax_enable_x64", True)

from tests.naive_shapley import naive_shapley

ATOL = 1e-6


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


def _check_single_game(phi_fn, a):
    a = np.asarray(a, dtype=np.float64)
    d = len(a)
    K = (a - 1.0).reshape(1, d)
    m_q = max(1, (d + 1) // 2)

    phi = phi_fn(K, m_q)
    assert phi.shape == (1, d)

    expected = naive_shapley(a)
    np.testing.assert_allclose(phi[0, :], expected, atol=ATOL)


def test_empty(phi_fn):
    K = np.empty((1, 0), dtype=np.float64)
    phi = phi_fn(K, m_q=1)
    assert phi.shape == (1, 0)


def test_simple(phi_fn):
    _check_single_game(phi_fn, [3.0])
    _check_single_game(phi_fn, [2.0, 3.0])
    _check_single_game(phi_fn, [2.0, -1.5])
    _check_single_game(phi_fn, [2.0, 0.0])

def test_random(phi_fn):
    rng = np.random.default_rng(42)
    for _ in range(1000):
        d = rng.integers(1, 8)
        a = rng.uniform(-10.0, 10.0, size=d)
        _check_single_game(phi_fn, a)


def _check_sum_property(phi_fn, m_q, a):
    a = np.asarray(a, dtype=np.float64)
    d = len(a)
    K = (a - 1.0).reshape(1, d)

    phi = phi_fn(K, m_q)
    phi_sum = phi[0, :].sum()
    expected_sum = np.prod(a) - 1.0
    np.testing.assert_allclose(phi_sum, expected_sum, rtol=1e-6)


@pytest.mark.parametrize("d,n_iter", [(100, 10), (1000, 10), (10000, 3)])
def test_sum_property_large(phi_fn, d, n_iter):
    # Check that sum of phi values equals prod(a) - 1.
    rng = np.random.default_rng(d)
    # Generate a = exp(uniform(-c, c)) so that log(prod(a)) stays bounded.
    # std(log(prod)) ≈ c * sqrt(d/3); choosing c = 5/sqrt(d) keeps it ≈ 2.9.
    c = 5.0 / np.sqrt(d)
    for _ in range(n_iter):
        a = np.exp(rng.uniform(-c, c, size=d))
        # Should be enough even for d=10000.
        m_q = 100
        _check_sum_property(phi_fn, m_q, a)
