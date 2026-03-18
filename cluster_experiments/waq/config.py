"""Configuration for WAQ density estimation and bootstrap inference."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class WAQConfig:
    """
    Parameters for WAQ point estimation and bootstrap inference.

    Attributes
    ----------
    density_sample_size
        If set, the control arm is subsampled (without replacement) to at most
        this many observations when estimating the density weights. Reduces cost
        for large controls. ``None`` uses all control units.
    bootstrap_samples
        Number of stratified bootstrap replications for SE, CI, and p-value.
    random_state
        Seed for subsampling and bootstrap.
    use_numba
        If True and numba is installed, use JIT-compiled KDE loops.
    """

    density_sample_size: Optional[int] = None
    bootstrap_samples: int = 399
    random_state: Optional[int] = None
    use_numba: bool = True
