# Weighted Average Quantile (WAQ) analysis

The **Weighted Average Quantile (WAQ)** estimator ([Athey, Bickel, Chen, Imbens & Pollmann, 2023](https://arxiv.org/abs/2109.02603)) targets the average treatment effect under a **constant additive (location-shift)** model on the outcome. It reweights quantile contrasts using the control density, which can **reduce variance** relative to difference-in-means for **thick-tailed** outcomes (e.g. revenue, GMV) when **CUPED / pre-period adjustment is not available**.

## When to use WAQ vs other methods

| Method | Role |
|--------|------|
| **CUPED** | Usually preferred when reliable pre-experiment outcomes exist. |
| **OLS / DiM** | Default mean effect; higher variance under heavy tails. |
| **WAQ** | Alternative when tails are heavy and pre-period data are scarce; assumes location-shift structure for the same estimand as the mean ATE. |

If the location-shift assumption fails, WAQ still estimates a **weighted combination of quantile effects**, which need not equal the difference in means.

## Scope in this package (v1)

- Intended for **simple unit-level A/B** (independent observations). **Not** for cluster-randomized designs without aggregating to the cluster level first; cluster-robust WAQ is not implemented.
- Inference uses **stratified bootstrap** (resample within arm) plus normal-based p-values from the bootstrap standard error.
- KDE uses a **triweight** kernel with sliding-window evaluation; install optional **Numba** for speed on large samples (`pip install numba` or `pip install 'cluster-experiments[performance]'` — **quote the extras in zsh** so `[performance]` is not treated as a glob).

## Usage

Use `analysis_type="waq"` in [`HypothesisTest`](api/hypothesis_test.md) with optional `analysis_config`:

- `density_sample_size`: cap control observations used to estimate weights (default: all controls).
- `bootstrap_samples`: bootstrap replications (default: 399).
- `random_state`: seed for reproducibility.
- `use_numba`: use Numba KDE when installed (default: `True`).
- `cluster_cols`: ignored (accepted for API compatibility).

See the tutorial notebook: [WAQ tutorial](examples/waq_tutorial.ipynb). For **Numba**, **JRSS-B §5-style simulations** (Normal / Laplace / Cauchy, Table 1), and a large-\\(n\\) timing demo, see [WAQ paper sims + scale](examples/waq_numba_scale.ipynb).

## References

- Paper: [arXiv:2109.02603](https://arxiv.org/abs/2109.02603)
- R code: [parTreat](https://github.com/michaelpollmann/parTreat)
- Feature request: [Issue #263](https://github.com/david26694/cluster-experiments/issues/263)
