"""
WAQ experiment analysis: simple A/B (independent units), not cluster-randomized.

Inference uses stratified bootstrap (resample within each arm) and normal
approximation for p-values from the bootstrap standard error.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd
from scipy.stats import norm

from cluster_experiments.experiment_analysis import (
    ConfidenceInterval,
    ExperimentAnalysis,
    InferenceResults,
)
from cluster_experiments.utils import HypothesisEntries
from cluster_experiments.waq.estimator import estimate_waq_tau


class WAQExperimentAnalysis(ExperimentAnalysis):
    """
    Weighted Average Quantile estimator (Athey et al., 2023).

    Intended for **simple randomized A/B** with independent units. Do **not**
    use for cluster-randomized experiments unless you aggregate to the cluster
    level first; v1 does not adjust inference for within-cluster correlation.

    Parameters
    ----------
    cluster_cols
        Ignored by WAQ; accepted for compatibility with :meth:`from_config`.
    density_sample_size
        Max control observations used to estimate density weights (full sample
        used for weighted means). ``None`` uses all controls.
    bootstrap_samples
        Stratified bootstrap replications for SE and CI.
    random_state
        RNG seed for bootstrap and subsampling.
    use_numba
        Use Numba-accelerated KDE when installed.

    References
    ----------
    Semiparametric Estimation of Treatment Effects in Randomized Experiments,
    https://arxiv.org/abs/2109.02603
    """

    def __init__(
        self,
        cluster_cols: Optional[List[str]] = None,
        target_col: str = "target",
        treatment_col: str = "treatment",
        treatment: str = "B",
        hypothesis: str = "two-sided",
        density_sample_size: Optional[int] = None,
        bootstrap_samples: int = 399,
        random_state: Optional[int] = None,
        use_numba: bool = True,
    ):
        super().__init__(
            cluster_cols=cluster_cols or [],
            target_col=target_col,
            treatment_col=treatment_col,
            treatment=treatment,
            covariates=None,
            hypothesis=hypothesis,
        )
        self.density_sample_size = density_sample_size
        self.bootstrap_samples = int(bootstrap_samples)
        self._waq_seed = random_state
        self.use_numba = use_numba

    def _split_y(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        t = df[self.treatment_col].to_numpy()
        y = df[self.target_col].to_numpy(dtype=np.float64)
        return y[t == 0], y[t == 1]

    def _run_bootstrap(
        self, y_c: np.ndarray, y_t: np.ndarray, rng: np.random.Generator
    ) -> np.ndarray:
        n_c, n_t = len(y_c), len(y_t)
        out = []
        for _ in range(self.bootstrap_samples):
            ic = rng.integers(0, n_c, size=n_c)
            it = rng.integers(0, n_t, size=n_t)
            try:
                tau_b, _ = estimate_waq_tau(
                    y_c[ic],
                    y_t[it],
                    density_sample_size=self.density_sample_size,
                    use_numba=self.use_numba,
                    rng=rng,
                )
                out.append(tau_b)
            except (ValueError, FloatingPointError):
                continue
        return np.asarray(out, dtype=np.float64)

    def _pvalue_from_z(self, tau: float, se: float) -> float:
        if se <= 0 or not np.isfinite(se):
            return 1.0
        z = tau / se
        h = HypothesisEntries(self.hypothesis)
        if h == HypothesisEntries.TWO_SIDED:
            return float(2.0 * min(norm.cdf(z), 1.0 - norm.cdf(z)))
        if h == HypothesisEntries.GREATER:
            return float(1.0 - norm.cdf(z))
        if h == HypothesisEntries.LESS:
            return float(norm.cdf(z))
        raise ValueError(self.hypothesis)

    def analysis_point_estimate(self, df: pd.DataFrame, verbose: bool = False) -> float:
        y_c, y_t = self._split_y(df)
        rng = np.random.default_rng(self._waq_seed)
        tau, _ = estimate_waq_tau(
            y_c,
            y_t,
            density_sample_size=self.density_sample_size,
            use_numba=self.use_numba,
            rng=rng,
        )
        return float(tau)

    def analysis_pvalue(self, df: pd.DataFrame, verbose: bool = False) -> float:
        return self.analysis_inference_results(df, alpha=0.05, verbose=verbose).p_value

    def analysis_standard_error(self, df: pd.DataFrame, verbose: bool = False) -> float:
        return self.analysis_inference_results(
            df, alpha=0.05, verbose=verbose
        ).std_error

    def analysis_confidence_interval(
        self, df: pd.DataFrame, alpha: float, verbose: bool = False
    ) -> ConfidenceInterval:
        return self.analysis_inference_results(
            df, alpha=alpha, verbose=verbose
        ).conf_int

    def analysis_inference_results(
        self, df: pd.DataFrame, alpha: float, verbose: bool = False
    ) -> InferenceResults:
        y_c, y_t = self._split_y(df)
        rng = np.random.default_rng(self._waq_seed)
        tau, _ = estimate_waq_tau(
            y_c,
            y_t,
            density_sample_size=self.density_sample_size,
            use_numba=self.use_numba,
            rng=rng,
        )
        boot_rng = np.random.default_rng(
            None if self._waq_seed is None else int(self._waq_seed) + 17
        )
        taus = self._run_bootstrap(y_c, y_t, boot_rng)
        if len(taus) < 10:
            se = 0.0
            lo, hi = float(tau), float(tau)
        else:
            se = float(np.std(taus, ddof=1))
            lo = float(np.quantile(taus, alpha / 2.0))
            hi = float(np.quantile(taus, 1.0 - alpha / 2.0))
        p = self._pvalue_from_z(float(tau), se)
        return InferenceResults(
            ate=float(tau),
            p_value=p,
            std_error=se,
            conf_int=ConfidenceInterval(lower=lo, upper=hi, alpha=alpha),
            fitted_model=None,
        )

    @classmethod
    def from_config(cls, config):
        return cls(
            cluster_cols=config.cluster_cols,
            target_col=config.target_col,
            treatment_col=config.treatment_col,
            treatment=config.treatment,
            hypothesis=config.hypothesis,
        )
