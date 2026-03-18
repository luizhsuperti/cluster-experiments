"""WAQ point estimator (triweight KDE on control, Athey et al. 2023)."""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np
from scipy.stats import norm

from cluster_experiments.waq.kde import kde_triweight


def _estimate_density_grid(
    X: np.ndarray, use_numba: bool
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Return xf (unique sorted), f, d2_log f at xf.

    Bandwidths follow the white-paper / parTreat-style rules.
    """
    dat = np.sort(np.asarray(X, dtype=np.float64).ravel())
    xf = np.unique(dat)
    n = len(dat)
    if n < 5:
        raise ValueError("WAQ requires at least 5 control observations")
    s = (np.quantile(dat, 0.95) - np.quantile(dat, 0.05)) / (2.0 * norm.ppf(0.95))
    if s <= 0:
        s = float(np.std(dat)) or 1.0
    h0 = s * 3.15 / (n ** (1.0 / 5.0))
    h2 = s * 2.70 / (n ** (1.0 / 9.0))
    f = kde_triweight(xf, dat, h0, 0, use_numba)
    f_d2 = kde_triweight(xf, dat, h2, 2, use_numba)
    f_d1 = np.gradient(f, xf)
    eps = 1e-12
    f_safe = np.maximum(f, eps)
    d2_logf = (f * f_d2 - f_d1**2) / (f_safe**2)
    return xf, f, d2_logf


def estimate_waq_tau(
    control: np.ndarray,
    treatment: np.ndarray,
    *,
    density_sample_size: Optional[int] = None,
    use_numba: bool = True,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[float, Dict[str, Any]]:
    """
    Weighted average quantile contrast: weighted mean(treatment Y)
    minus weighted mean(control Y) with weights from control's -d² log f.

    Parameters
    ----------
    control
        Outcomes in control arm.
    treatment
        Outcomes in treatment arm.
    density_sample_size
        Subsample size for KDE (full arms still used for weighted means).
    use_numba
        Use Numba KDE when available.
    rng
        Required if density_sample_size is set and len(control) exceeds it.

    Returns
    -------
    tau
        Point estimate.
    info
        Diagnostics (xf grid size, etc.).
    """
    control = np.asarray(control, dtype=np.float64).ravel()
    treatment = np.asarray(treatment, dtype=np.float64).ravel()
    if len(treatment) < 1 or len(control) < 5:
        raise ValueError("WAQ needs at least 5 control and 1 treatment observation")

    dat_density = control
    if density_sample_size is not None and len(control) > density_sample_size:
        if rng is None:
            rng = np.random.default_rng()
        pick = rng.choice(len(control), size=density_sample_size, replace=False)
        dat_density = control[pick]

    xf, _f, d2_logf = _estimate_density_grid(dat_density, use_numba)
    w = -d2_logf
    valid = np.isfinite(w) & np.isfinite(xf)
    xf_c, w_c = xf[valid], w[valid]
    if len(xf_c) < 3:
        raise ValueError("WAQ: insufficient valid density weights")

    control_s = np.sort(control)
    treatment_s = np.sort(treatment)
    w_co = np.interp(control_s, xf_c, w_c)
    w_tr = np.interp(treatment_s, xf_c, w_c)
    w_co = np.maximum(w_co, 1e-12)
    w_tr = np.maximum(w_tr, 1e-12)

    mean_t = float(np.sum(w_tr * treatment_s) / np.sum(w_tr))
    mean_c = float(np.sum(w_co * control_s) / np.sum(w_co))
    tau = mean_t - mean_c
    info = {
        "n_control": len(control),
        "n_treatment": len(treatment),
        "n_density": len(dat_density),
        "n_grid": len(xf_c),
    }
    return tau, info


class WAQEstimator:
    """Thin wrapper around :func:`estimate_waq_tau`."""

    def __init__(
        self,
        density_sample_size: Optional[int] = None,
        use_numba: bool = True,
        random_state: Optional[int] = None,
    ):
        self.density_sample_size = density_sample_size
        self.use_numba = use_numba
        self.random_state = random_state

    def fit(
        self,
        control: np.ndarray,
        treatment: np.ndarray,
    ) -> Tuple[float, Dict[str, Any]]:
        rng = (
            np.random.default_rng(self.random_state)
            if self.random_state is not None
            else None
        )
        return estimate_waq_tau(
            control,
            treatment,
            density_sample_size=self.density_sample_size,
            use_numba=self.use_numba,
            rng=rng,
        )
