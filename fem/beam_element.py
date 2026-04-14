"""
fem/beam_element.py
Euler-Bernoulli beam element (2 DOF/node: transverse w, rotation θ).
"""
import numpy as np
import config


def stiffness(E, I, Le):
    """
    4×4 Euler-Bernoulli beam element stiffness matrix.
    DOF order: [w_i, θ_i, w_j, θ_j]
    """
    k = E * I / Le ** 3
    L = Le
    return k * np.array([
        [12,    6*L,   -12,    6*L],
        [6*L,   4*L**2, -6*L,  2*L**2],
        [-12,  -6*L,    12,   -6*L],
        [6*L,   2*L**2, -6*L,  4*L**2]
    ])


def consistent_mass(rho, A, Le):
    """4×4 consistent mass matrix (Euler-Bernoulli)."""
    m = rho * A * Le / 420
    L = Le
    return m * np.array([
        [156,    22*L,    54,   -13*L],
        [22*L,   4*L**2,  13*L, -3*L**2],
        [54,     13*L,   156,   -22*L],
        [-13*L, -3*L**2, -22*L,  4*L**2]
    ])


def electrostrictive_equiv_force(eps_e, E, I, Le, kind='uniform'):
    """
    Equivalent nodal force vector for an electrostrictive actuator bonded
    to the **top surface** of the beam.

    The actuator introduces a prescribed surface strain ε_e at z = +h/2.
    This is equivalent to imposing an eigen-curvature κ_e = ε_e / (h/2).
    But since h = t and I = b·t³/12,  A = b·t,  c = t/2:

        κ_e = 2 ε_e / t

    The consistent equivalent nodal force vector from virtual work is:
        f_eq = ∫₀ᴸ B(x)^T · (E I κ_e) dx

    For uniform κ_e and the Euler-Bernoulli B matrix this integrates to:
        f_eq = E I κ_e · [0, 1, 0, -1]   (two moment-DOF components)

    This is the standard result used in piezoelectric/thermal FEM.

    Parameters
    ----------
    eps_e : electrostrictive surface strain (top fibre), ε = M(V/t)²
    E, I  : material / section properties
    Le    : element length (unused in this formula but kept for API compat)
    """
    # Curvature from top-surface strain, using c = t/2 → I/A = t²/12 → c = √(3 I / (b t))
    # For simplicity treat it as a known bending moment per element
    M_bend = E * I * eps_e * (2.0 / config.h)    # E·I·κ_e
    return np.array([0.0, -M_bend, 0.0, M_bend])   # drives beam upward (+w)


def shape_functions(xi, Le):
    """
    Hermite shape functions evaluated at parametric coord ξ ∈ [-1,1].
    Returns N (1×4).
    """
    L = Le
    N1 = 0.25 * (1 - xi) ** 2 * (2 + xi)
    N2 = 0.25 * L * (1 - xi) ** 2 * (1 + xi)
    N3 = 0.25 * (1 + xi) ** 2 * (2 - xi)
    N4 = 0.25 * L * (1 + xi) ** 2 * (xi - 1)
    return np.array([N1, N2, N3, N4])


def strain_displacement(xi, Le):
    """
    Curvature-displacement matrix B (second derivative of shape functions).
    κ = B u → ε_xx = -z κ
    """
    L = Le
    x = (xi + 1) / 2 * L   # physical coord
    d2N1 = (6 * x / L ** 3 - 6 / (2 * L ** 2)) * 2   # chain rule jacobian
    # Use direct physical derivatives
    x_ = x
    B1 =  6 / L ** 3 * (2 * x_ - L) / L
    B2 =  2 / L ** 2 * (3 * x_ / L - 2)
    B3 = -6 / L ** 3 * (2 * x_ - L) / L
    B4 =  2 / L ** 2 * (3 * x_ / L - 1)
    return np.array([B1, B2, B3, B4])
