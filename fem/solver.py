"""
fem/solver.py
Linear static solver, modal solver, and stress recovery for beams.
"""
import numpy as np
from scipy.linalg import solve, eigh


# ------------------------------------------------------------------ #
#  Static solver                                                       #
# ------------------------------------------------------------------ #

def solve_static(K_ff, F_f, free_dofs, ndof):
    """
    Solve K_ff u_f = F_f and reconstruct full displacement vector.
    Returns u (ndof,).
    """
    u_f = np.linalg.solve(K_ff, F_f)
    u = np.zeros(ndof)
    u[free_dofs] = u_f
    return u


# ------------------------------------------------------------------ #
#  Modal solver (generalised eigenvalue)                              #
# ------------------------------------------------------------------ #

def solve_modal(K_ff, M_ff, n_modes=6):
    """
    Solve K u = λ M u  (generalised eigenvalue problem).
    Returns:
      freqs_hz : natural frequencies [Hz]
      modes    : mode shape columns (free DOFs), shape (ndof_free, n_modes)
    """
    n_modes = min(n_modes, K_ff.shape[0])
    eigenvalues, eigenvectors = eigh(K_ff, M_ff, subset_by_index=[0, n_modes - 1])
    eigenvalues = np.maximum(eigenvalues, 0.0)   # numerical safety
    freqs_hz = np.sqrt(eigenvalues) / (2 * np.pi)
    return freqs_hz, eigenvectors


# ------------------------------------------------------------------ #
#  Stress / strain recovery along beam                                #
# ------------------------------------------------------------------ #

def recover_beam_stress(u, n_elem, Le, E, I, h, eps_e_array=None):
    """
    Recover bending stress σ_xx = -(E z κ) at top fibre (z = h/2) for each element.

    Returns
    -------
    x_mid  : element mid-point x coordinates
    stress : bending stress at top fibre for each element
    kappa  : curvature at mid of each element
    """
    x_mid  = np.zeros(n_elem)
    stress = np.zeros(n_elem)
    kappa  = np.zeros(n_elem)

    E_arr = np.broadcast_to(E, (n_elem,))
    I_arr = np.broadcast_to(I, (n_elem,))

    for e in range(n_elem):
        idx = [2*e, 2*e+1, 2*e+2, 2*e+3]
        u_e = u[idx]
        L = Le
        # Curvature at element midpoint using Euler-Bernoulli B matrix
        # d²/dx² of Hermite shape functions at x = L/2
        x_ = L / 2
        # B = [d²N₁/dx², d²N₂/dx², d²N₃/dx², d²N₄/dx²]
        B = np.array([
             12*x_/L**3 - 6/L**2,
              6*x_/L**2 - 4/L,
             -12*x_/L**3 + 6/L**2,
              6*x_/L**2 - 2/L
        ])
        k_e = B @ u_e
        kappa[e]  = k_e
        eps_e = eps_e_array[e] if eps_e_array is not None else 0.0
        # σ = -E z κ + E ε_axial; top fibre z = +h/2 → tension when κ>0
        sigma_e = E_arr[e] * (k_e * (h / 2) - eps_e)   # net top-fibre stress
        stress[e] = sigma_e
        x_mid[e]  = (e + 0.5) * Le

    return x_mid, stress, kappa
