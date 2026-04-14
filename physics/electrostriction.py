"""
physics/electrostriction.py
Electrostrictive strain computations.
"""
import numpy as np


def electro_strain(V, t, M):
    """
    Uniform electrostrictive strain: ε = M (V/t)²
    """
    return M * (V / t) ** 2


def spatial_electro_strain(x, L, V, t, M):
    """
    Spatially varying electrostrictive strain (sinusoidal distribution along beam).
    ε(x) = M (V/t)² sin(πx/L)
    """
    return M * (V / t) ** 2 * np.sin(np.pi * x / L)


def electro_strain_gradient(V, t, M, L, n_points=100):
    """
    Return (x, ε(x)) arrays for the sinusoidal distribution over the beam.
    """
    x = np.linspace(0, L, n_points)
    eps = spatial_electro_strain(x, L, V, t, M)
    return x, eps


def electrostrictive_energy(V, t, M, volume):
    """
    Stored electrostrictive energy density: u = ½ ε² / M  (simplified)
    U = u * volume
    """
    eps = electro_strain(V, t, M)
    # Using: σ·ε / 2 analogy; here we store as ε² * E / 2
    return eps ** 2 * volume / (2 * M + 1e-30)
