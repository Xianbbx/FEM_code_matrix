"""
fem/assembly.py
Global stiffness / mass / force assembly for beam FEM.
"""
import numpy as np
from fem.beam_element import stiffness, consistent_mass, electrostrictive_equiv_force


def assemble_beam(n_elem, Le, E, I, rho=None, A=None,
                  eps_e_array=None, kind='uniform'):
    """
    Assemble global K (and optionally M) for a beam with n_elem elements.

    Parameters
    ----------
    n_elem      : number of elements
    Le          : element length
    E           : Young's modulus (scalar or array length n_elem)
    I           : second moment of area (scalar or array length n_elem)
    rho         : density (required for mass matrix)
    A           : cross-sectional area (required for mass matrix)
    eps_e_array : electrostrictive strain per element (None → F=0)
    kind        : 'uniform' force distribution

    Returns
    -------
    K : (ndof × ndof)
    M : (ndof × ndof) or None
    F : (ndof,)
    """
    n_nodes = n_elem + 1
    ndof = 2 * n_nodes

    K = np.zeros((ndof, ndof))
    M_g = np.zeros((ndof, ndof)) if (rho is not None and A is not None) else None
    F = np.zeros(ndof)

    E_arr = np.broadcast_to(E, (n_elem,))
    I_arr = np.broadcast_to(I, (n_elem,))

    for e in range(n_elem):
        ke = stiffness(E_arr[e], I_arr[e], Le)
        idx = [2*e, 2*e+1, 2*e+2, 2*e+3]

        for i in range(4):
            for j in range(4):
                K[idx[i], idx[j]] += ke[i, j]

        if M_g is not None:
            rho_e = rho[e] if hasattr(rho, '__len__') else rho
            A_e   = A[e]   if hasattr(A,   '__len__') else A
            me = consistent_mass(rho_e, A_e, Le)
            for i in range(4):
                for j in range(4):
                    M_g[idx[i], idx[j]] += me[i, j]

        if eps_e_array is not None:
            fe = electrostrictive_equiv_force(eps_e_array[e], E_arr[e], I_arr[e], Le, kind)
            for i in range(4):
                F[idx[i]] += fe[i]

    return K, M_g, F


def apply_bc(K, F, fixed_dofs, M=None):
    """
    Apply homogeneous Dirichlet BCs by partitioning.

    Returns
    -------
    K_ff, F_f, free_dofs  (and M_ff if M provided)
    """
    ndof = K.shape[0]
    fixed_dofs = list(fixed_dofs)
    free_dofs  = [d for d in range(ndof) if d not in fixed_dofs]

    K_ff = K[np.ix_(free_dofs, free_dofs)]
    F_f  = F[free_dofs]
    M_ff = M[np.ix_(free_dofs, free_dofs)] if M is not None else None

    return K_ff, F_f, free_dofs, M_ff
