"""
physics/elasticity.py
Linear elastic constitutive relations.
"""
import numpy as np


def D_matrix_2D_plane_stress(E, nu):
    """
    3×3 constitutive matrix for plane stress.
    Voigt notation: [εxx, εyy, γxy] → [σxx, σyy, σxy]
    """
    c = E / (1 - nu ** 2)
    return c * np.array([
        [1,  nu, 0],
        [nu, 1,  0],
        [0,  0,  (1 - nu) / 2]
    ])


def D_matrix_2D_plane_strain(E, nu):
    """3×3 constitutive matrix for plane strain."""
    c = E / ((1 + nu) * (1 - 2 * nu))
    return c * np.array([
        [1 - nu,  nu,       0],
        [nu,      1 - nu,   0],
        [0,       0,        (1 - 2 * nu) / 2]
    ])


def stress_from_strain_1D(E, eps_total, eps_eigen=0.0):
    """
    σ = E (ε_total - ε_eigen)
    """
    return E * (eps_total - eps_eigen)


def stress_from_strain_voigt(D, eps_voigt, eps_eigen=None):
    """
    σ = D (ε - ε_eigen)   (Voigt)
    D : (n×n),  eps_voigt : (n,)
    """
    if eps_eigen is None:
        eps_eigen = np.zeros_like(eps_voigt)
    return D @ (eps_voigt - eps_eigen)


def principal_stresses_2D(sxx, syy, sxy):
    """
    Analytical principal stresses for 2-D state.
    Returns (σ₁, σ₂).
    """
    avg = (sxx + syy) / 2
    rad = np.sqrt(((sxx - syy) / 2) ** 2 + sxy ** 2)
    return avg + rad, avg - rad


def principal_stresses_3D(stress_tensor_3x3):
    """
    Return eigenvalues (σ₁ ≥ σ₂ ≥ σ₃) of a symmetric 3×3 stress tensor.
    """
    vals = np.linalg.eigvalsh(stress_tensor_3x3)
    return np.sort(vals)[::-1]
