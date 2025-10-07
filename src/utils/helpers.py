"""Helper utilities for data processing"""
import numpy as np

def safe_dt(t):
    """Compute safe time deltas"""
    t = np.asarray(t, float)
    dt = np.diff(t, prepend=t[0])
    pos = dt[np.isfinite(dt) & (dt > 0)]
    fill = np.median(pos) if pos.size else 0.1
    dt[~np.isfinite(dt) | (dt <= 0)] = fill
    return dt

def ffill_2d(arr):
    """Forward fill NaN values in 2D array"""
    a = np.asarray(arr, float).copy()
    finite = np.isfinite(a).all(axis=1)
    if not finite.any():
        return np.nan_to_num(a, nan=0.0)
    last = np.where(finite)[0][0]
    if last > 0:
        a[:last] = a[last]
    for i in range(last + 1, len(a)):
        if not np.isfinite(a[i]).all():
            a[i] = a[i - 1]
    return a

def clamp_vec_norm(v, max_mag):
    """Clamp vector magnitude"""
    n = np.linalg.norm(v, axis=-1, keepdims=True) + 1e-9
    scale = np.minimum(1.0, max_mag / n)
    return v * scale
