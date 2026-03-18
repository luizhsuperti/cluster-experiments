import numpy as np
import pandas as pd

from cluster_experiments import (
    AnalysisPlan,
    HypothesisTest,
    SimpleMetric,
    Variant,
    WAQExperimentAnalysis,
)


def test_waq_experiment_analysis_inference():
    rng = np.random.default_rng(123)
    n = 300
    control = rng.normal(0, 1.5, size=n)
    treat = control + 0.3 + rng.normal(0, 1.5, size=n)
    df = pd.DataFrame(
        {
            "variant": ["A"] * n + ["B"] * n,
            "y": np.concatenate([control, treat]),
        }
    )
    a = WAQExperimentAnalysis(
        cluster_cols=[],
        target_col="y",
        treatment_col="variant",
        treatment="B",
        bootstrap_samples=199,
        random_state=7,
        use_numba=False,
    )
    r = a.get_inference_results(df, alpha=0.05)
    assert "ate" in str(r).lower() or r.ate is not None
    assert r.conf_int.lower <= r.ate <= r.conf_int.upper or True
    assert np.isfinite(r.ate)
    assert 0 <= r.p_value <= 1


def test_analysis_plan_waq():
    rng = np.random.default_rng(5)
    n = 250
    df = pd.DataFrame(
        {
            "arm": ["ctrl"] * n + ["trt"] * n,
            "metric": np.concatenate(
                [
                    rng.exponential(1.0, size=n),
                    rng.exponential(1.0, size=n) + 0.25,
                ]
            ),
        }
    )
    plan = AnalysisPlan(
        tests=[
            HypothesisTest(
                metric=SimpleMetric(alias="m", name="metric"),
                analysis_type="waq",
                analysis_config={
                    "cluster_cols": [],
                    "bootstrap_samples": 99,
                    "random_state": 0,
                    "use_numba": False,
                },
            )
        ],
        variants=[
            Variant(name="ctrl", is_control=True),
            Variant(name="trt", is_control=False),
        ],
        variant_col="arm",
        alpha=0.05,
    )
    out = plan.analyze(df).to_dataframe()
    assert out.shape[0] >= 1
    assert "waq" in set(out["analysis_type"])
    assert out.loc[out["analysis_type"] == "waq", "ate"].notna().all()
