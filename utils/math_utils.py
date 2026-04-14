"""
utils/math_utils.py
General mathematical helpers used across modules.
"""
import numpy as np


def norm_vector(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-15 else v


def smooth_step(x, x0, x1):
    """Smooth Hermite interpolation between 0 and 1 over [x0, x1]."""
    t = np.clip((x - x0) / (x1 - x0 + 1e-30), 0, 1)
    return t * t * (3 - 2 * t)


def running_max(arr):
    """Element-wise cumulative maximum."""
    out = np.empty_like(arr)
    out[0] = arr[0]
    for i in range(1, len(arr)):
        out[i] = max(out[i-1], arr[i])
    return out


def interp_field(x_src, f_src, x_dst):
    """Linear interpolation of field f from x_src to x_dst."""
    return np.interp(x_dst, x_src, f_src)


def rms(arr):
    return np.sqrt(np.mean(arr ** 2))
