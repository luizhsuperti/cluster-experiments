import numpy as np
import pytest

from cluster_experiments.waq.estimator import estimate_waq_tau


def test_identical_arms_near_zero():
    rng = np.random.default_rng(0)
    y = rng.lognormal(mean=1.0, sigma=0.5, size=500)
    tau, _ = estimate_waq_tau(y, y, use_numba=False, rng=rng)
    assert abs(tau) < 0.15


def test_location_shift_normal_close_to_mean_diff():
    rng = np.random.default_rng(42)
    c = rng.normal(0, 1, size=800)
    t = c[:400].copy() + 0.5  # not iid but similar marginals
    t = np.concatenate([t, rng.normal(0.5, 1, size=400)])
    tau, _ = estimate_waq_tau(c, t, use_numba=False, rng=rng)
    dim = float(np.mean(t) - np.mean(c))
    assert abs(tau - dim) < 0.08


def test_subsample_density():
    rng = np.random.default_rng(1)
    c = rng.exponential(1.0, size=2000)
    t = c[:1000] + 0.2
    t = np.concatenate([t, rng.exponential(1.0, size=1000) + 0.2])
    tau_full, _ = estimate_waq_tau(
        c, t, density_sample_size=None, use_numba=False, rng=rng
    )
    rng2 = np.random.default_rng(1)
    tau_sub, _ = estimate_waq_tau(
        c, t, density_sample_size=400, use_numba=False, rng=rng2
    )
    assert np.isfinite(tau_full) and np.isfinite(tau_sub)


def test_too_few_control_raises():
    with pytest.raises(ValueError, match="at least 5 control"):
        estimate_waq_tau(np.array([1.0, 2.0]), np.array([1.0]), use_numba=False)
