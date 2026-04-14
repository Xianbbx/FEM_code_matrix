"""
physics/plasticity.py
Von-Mises yield criterion and related utilities.
"""
import numpy as np


def von_mises_from_principals(s1, s2, s3=0.0):
    """
    σ_vm = √( ½ [(σ1-σ2)² + (σ2-σ3)² + (σ3-σ1)²] )
    Handles 2-D by defaulting s3 = 0.
    """
    return np.sqrt(0.5 * ((s1 - s2) ** 2 +
                          (s2 - s3) ** 2 +
                          (s3 - s1) ** 2))


def von_mises_from_components(sxx, syy, szz=0.0, sxy=0.0, syz=0.0, sxz=0.0):
    """
    Full Von-Mises from stress components.
    """
    return np.sqrt(0.5 * (
        (sxx - syy) ** 2 +
        (syy - szz) ** 2 +
        (szz - sxx) ** 2 +
        6 * (sxy ** 2 + syz ** 2 + sxz ** 2)
    ))


def check_yield(sigma_vm, sigma_y):
    """Return boolean mask where plasticity occurs."""
    return np.asarray(sigma_vm) >= sigma_y


def yield_surface_cylinder(sigma_y, n_theta=200):
    """
    Parametric Von-Mises yield cylinder in (σ₁, σ₂, σ₃) space
    (restricted to the π-plane: σ₁+σ₂+σ₃ = 0).
    Returns (x, y, z) each shape (n_theta, n_z) for a surface plot.
    """
    # The cylinder axis is along σ_hydrostatic direction [1,1,1]/√3
    e1 = np.array([1, -1,  0]) / np.sqrt(2)    # basis on π-plane
    e2 = np.array([1,  1, -2]) / np.sqrt(6)    # second basis

    theta = np.linspace(0, 2 * np.pi, n_theta)
    z_vals = np.linspace(-1.5 * sigma_y, 1.5 * sigma_y, 4) / np.sqrt(3)

    pts = sigma_y * np.sqrt(2 / 3) * (
        np.outer(np.cos(theta), e1) + np.outer(np.sin(theta), e2)
    )  # shape (n_theta, 3) — on π-plane at σ_h = 0

    # Extrude along hydrostatic axis
    hydro_axis = np.array([1, 1, 1]) / np.sqrt(3)
    cyl = []
    for zh in z_vals:
        cyl.append(pts + zh * hydro_axis)
    # shape (n_z, n_theta, 3)
    cyl = np.array(cyl)
    return cyl[:, :, 0], cyl[:, :, 1], cyl[:, :, 2]


def yield_locus_pi_plane(sigma_y, n_theta=300):
    """
    Return (ξ, η) coordinates of the Von-Mises circle on the π-plane.
    """
    theta = np.linspace(0, 2 * np.pi, n_theta)
    r = sigma_y * np.sqrt(2.0 / 3.0)
    return r * np.cos(theta), r * np.sin(theta)


def distortion_energy(s1, s2, s3):
    """
    Distortion energy density (normalised by shear modulus):
    φ_d = (1/6G) [(σ1-σ2)² + (σ2-σ3)² + (σ3-σ1)²]
    Returns the bracket term (caller handles /6G).
    """
    return (s1 - s2) ** 2 + (s2 - s3) ** 2 + (s3 - s1) ** 2


def volumetric_deviatoric_split(sxx, syy, szz):
    """
    Return (σ_vol, s_dev) where:
      σ_vol = (1/3) tr(σ)
      s_dev = [sxx-σ_vol, syy-σ_vol, szz-σ_vol]
    """
    sv = (sxx + syy + szz) / 3.0
    return sv, np.array([sxx - sv, syy - sv, szz - sv])
