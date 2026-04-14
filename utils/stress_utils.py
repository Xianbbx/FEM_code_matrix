"""
utils/stress_utils.py
Stress tensor decomposition and derived quantities.
"""
import numpy as np
from physics.plasticity import von_mises_from_principals
from physics.elasticity  import principal_stresses_2D


def volumetric_deviatoric_3d(sxx, syy, szz, sxy=0, syz=0, sxz=0):
    """
    Full 3-D volumetric / deviatoric decomposition.
    Returns scalar σ_vol and deviatoric component array.
    """
    sv = (sxx + syy + szz) / 3.0
    s_dev = np.array([sxx - sv,
                      syy - sv,
                      szz - sv,
                      sxy,
                      syz,
                      sxz])
    return sv, s_dev


def beam_stress_along_depth(sigma_bending, sigma_axial, z_points, h):
    """
    Stress distribution through beam thickness (linear bending + uniform axial).
    z_points : array of z coordinates (0 = neutral axis)
    """
    return sigma_axial + sigma_bending * (z_points / (h / 2))


def stress_mohr_circle_2d(sxx, syy, sxy):
    """
    Compute Mohr's circle parameters.
    Returns centre, radius, principal angles.
    """
    center = (sxx + syy) / 2.0
    radius = np.sqrt(((sxx - syy) / 2) ** 2 + sxy ** 2)
    angle  = 0.5 * np.arctan2(2 * sxy, sxx - syy)
    return center, radius, np.degrees(angle)


def mohr_circle_points(center, radius, n=200):
    """Return (σ_n, τ) pairs on the Mohr's circle."""
    theta = np.linspace(0, 2 * np.pi, n)
    s_n = center + radius * np.cos(theta)
    tau = radius * np.sin(theta)
    return s_n, tau


def principal_from_beam_state(sigma_bending, sigma_axial=0.0):
    """
    For a uniaxial bending state: σ_1 = σ_bending + σ_axial, σ_2 = σ_3 = 0.
    """
    s1 = sigma_bending + sigma_axial
    return s1, 0.0, 0.0


def von_mises_beam(sigma_bending, tau_shear=0.0):
    """
    Von-Mises for beam: σ_vm = √(σ² + 3τ²)
    """
    return np.sqrt(sigma_bending ** 2 + 3 * tau_shear ** 2)
