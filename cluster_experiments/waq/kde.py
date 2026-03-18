"""Triweight KDE on a sorted grid with O(n + m) sliding-window evaluation."""

from __future__ import annotations

import numpy as np

# Triweight kernel K(u) and derivatives, |u| <= 1 (Biweight family / R parTreat)


def _triweight_u(u: float, derivative: int) -> float:
    au = abs(u)
    if au > 1.0:
        return 0.0
    if derivative == 0:
        return (35.0 / 32.0) * (1.0 - u * u) ** 3
    if derivative == 2:
        return -(105.0 / 16.0) * (1.0 - 5.0 * u * u) * (1.0 - u * u)
    raise ValueError("derivative must be 0 or 2")


def kde_triweight_sorted(
    xf: np.ndarray,
    dat: np.ndarray,
    h: float,
    derivative: int,
) -> np.ndarray:
    """
    KDE mean: (1/n) sum_j K_d((x - x_j)/h) / h^(d+1).

    Parameters
    ----------
    xf
        Evaluation points (sorted ascending).
    dat
        Data used in KDE (sorted ascending).
    h
        Positive bandwidth.
    derivative
        0 for density, 2 for second derivative of density.
    """
    if h <= 0:
        raise ValueError("bandwidth h must be positive")
    n = len(dat)
    m = len(xf)
    out = np.empty(m, dtype=np.float64)
    left = 0
    right = -1
    for i in range(m):
        x = float(xf[i])
        lo = x - h
        hi = x + h
        while left < n and float(dat[left]) < lo:
            left += 1
        if right < left - 1:
            right = left - 1
        while right + 1 < n and float(dat[right + 1]) <= hi:
            right += 1
        acc = 0.0
        for j in range(left, right + 1):
            u = (x - float(dat[j])) / h
            acc += _triweight_u(u, derivative)
        out[i] = acc / (h ** (derivative + 1) * n)
    return out


def try_import_numba_kde():
    """Return JIT kde function or None."""
    try:
        import numba

        @numba.njit(cache=True)
        def _triweight_u_nb(u: float, derivative: int) -> float:
            au = abs(u)
            if au > 1.0:
                return 0.0
            if derivative == 0:
                return (35.0 / 32.0) * (1.0 - u * u) ** 3
            if derivative == 2:
                return -(105.0 / 16.0) * (1.0 - 5.0 * u * u) * (1.0 - u * u)
            return 0.0

        @numba.njit(cache=True)
        def kde_triweight_numba(
            xf: np.ndarray, dat: np.ndarray, h: float, derivative: int
        ) -> np.ndarray:
            n = len(dat)
            m = len(xf)
            out = np.empty(m, dtype=np.float64)
            left = 0
            right = -1
            hd = h ** (derivative + 1) * n
            for i in range(m):
                x = xf[i]
                lo = x - h
                hi = x + h
                while left < n and dat[left] < lo:
                    left += 1
                if right < left - 1:
                    right = left - 1
                while right + 1 < n and dat[right + 1] <= hi:
                    right += 1
                acc = 0.0
                for j in range(left, right + 1):
                    u = (x - dat[j]) / h
                    acc += _triweight_u_nb(u, derivative)
                out[i] = acc / hd
            return out

        return kde_triweight_numba
    except Exception:
        return None


_NUMBA_KDE = None


def get_numba_kde():
    global _NUMBA_KDE
    if _NUMBA_KDE is False:
        return None
    if _NUMBA_KDE is None:
        _NUMBA_KDE = try_import_numba_kde() or False
    return _NUMBA_KDE if _NUMBA_KDE is not False else None


def kde_triweight(
    xf: np.ndarray,
    dat: np.ndarray,
    h: float,
    derivative: int,
    use_numba: bool,
) -> np.ndarray:
    xf = np.asarray(xf, dtype=np.float64)
    dat = np.asarray(dat, dtype=np.float64)
    if use_numba:
        nb = get_numba_kde()
        if nb is not None:
            return nb(xf, dat, float(h), int(derivative))
    return kde_triweight_sorted(xf, dat, h, derivative)


def _triweight_array(u: np.ndarray, derivative: int) -> np.ndarray:
    m = np.abs(u) <= 1.0
    if derivative == 0:
        return np.where(m, (35.0 / 32.0) * (1.0 - u * u) ** 3, 0.0)
    if derivative == 2:
        return np.where(m, -(105.0 / 16.0) * (1.0 - 5.0 * u * u) * (1.0 - u * u), 0.0)
    raise ValueError("derivative must be 0 or 2")


def kde_triweight_naive(
    xf: np.ndarray, dat: np.ndarray, h: float, derivative: int
) -> np.ndarray:
    """O(m * n) dense reference for tests."""
    u = (xf[:, None] - dat[None, :]) / h
    k = _triweight_array(u, derivative)
    return k.mean(axis=1) / (h ** (derivative + 1))
