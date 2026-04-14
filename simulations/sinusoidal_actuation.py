"""
simulations/sinusoidal_actuation.py
Intermediate Simulation: Spatial sinusoidal electrostrictive strain.
ε(x) = M (V/t)² sin(πx/L)
Shows wavelike deformation patterns + evolution animation.
"""
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from physics.electrostriction import spatial_electro_strain
from fem.beam_element import stiffness, electrostrictive_equiv_force
from fem.assembly import apply_bc
from fem.solver import solve_static
from visualization import plots, animations


def run_sinusoidal_fem(V):
    """Solve beam FEM with spatially varying (sinusoidal) electrostrictive load."""
    L     = config.L
    n_elem = config.n_elem
    Le    = L / n_elem
    ndof  = 2 * (n_elem + 1)

    K = np.zeros((ndof, ndof))
    F = np.zeros(ndof)

    for e in range(n_elem):
        ke = stiffness(config.E, config.I, Le)
        idx = [2*e, 2*e+1, 2*e+2, 2*e+3]
        K[np.ix_(idx, idx)] += ke

        # Evaluate strain at element midpoint
        x_mid = (e + 0.5) * Le
        eps_e = spatial_electro_strain(x_mid, L, V, config.t, config.M)
        fe = electrostrictive_equiv_force(eps_e, config.E, config.I, Le)
        F[np.array(idx)] += fe

    K_ff, F_f, free_dofs, _ = apply_bc(K, F, fixed_dofs=[0, 1])
    u = solve_static(K_ff, F_f, free_dofs, ndof)

    x = np.linspace(0, L, n_elem + 1)
    w = u[0::2]
    return x, w


def run():
    print('\n=== INTERMEDIATE SIM: Sinusoidal Actuation ===')

    voltages = np.linspace(10, config.V_max, 20)
    w_list = []
    for V in voltages:
        x, w = run_sinusoidal_fem(V)
        w_list.append(w)

    plots.plot_sinusoidal_deformation(x, w_list, voltages,
                                      tag='11_sinusoidal_deformation')

    # Animation
    def get_w_sin(V):
        _, ws = run_sinusoidal_fem(V)
        return ws

    v_anim = np.linspace(0, config.V_max, 60)
    animations.animate_sinusoidal_actuation(x, get_w_sin, v_anim,
                                             tag='02_sinusoidal_anim')
    print('  [DONE] sinusoidal_actuation simulation')


if __name__ == '__main__':
    run()
