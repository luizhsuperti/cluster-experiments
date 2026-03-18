import numpy as np
import pytest

from cluster_experiments.waq.kde import (
    get_numba_kde,
    kde_triweight,
    kde_triweight_naive,
    kde_triweight_sorted,
)


@pytest.mark.parametrize("derivative", [0, 2])
@pytest.mark.parametrize("seed", [0, 1, 42])
def test_sparse_matches_naive(derivative, seed):
    rng = np.random.default_rng(seed)
    dat = np.sort(rng.normal(size=80))
    xf = np.sort(np.unique(dat))
    h = 0.35
    dense = kde_triweight_naive(xf, dat, h, derivative)
    sparse = kde_triweight_sorted(xf, dat, h, derivative)
    np.testing.assert_allclose(sparse, dense, rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize("derivative", [0, 2])
def test_numba_matches_numpy(derivative):
    nb = get_numba_kde()
    if nb is None:
        pytest.skip("numba not installed")
    rng = np.random.default_rng(7)
    dat = np.sort(rng.exponential(scale=2.0, size=100))
    xf = np.sort(np.unique(dat))[:50]
    h = 0.5
    ref = kde_triweight_sorted(xf, dat, h, derivative)
    got = nb(xf, dat, h, derivative)
    np.testing.assert_allclose(got, ref, rtol=1e-10, atol=1e-11)


def test_kde_triweight_use_numba_flag():
    rng = np.random.default_rng(0)
    dat = np.sort(rng.normal(size=60))
    xf = np.sort(np.unique(dat))[:40]
    h = 0.4
    a = kde_triweight(xf, dat, h, 0, use_numba=False)
    b = kde_triweight(xf, dat, h, 0, use_numba=True)
    np.testing.assert_allclose(a, b, rtol=1e-10, atol=1e-11)
