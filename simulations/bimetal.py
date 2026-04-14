"""
simulations/bimetal.py
Intermediate Simulation: Bimetal strip with E-mismatch.
Two-layer composite: E1 (Al) top / E2 (Steel) bottom.
Mismatch strain (like thermal) drives bending.
"""
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from fem.beam_element import stiffness, consistent_mass, electrostrictive_equiv_force
from fem.assembly import apply_bc
from fem.solver import solve_static, recover_beam_stress
from visualization import plots


def _bimetal_stiffness(E_eff, I_eff, Le):
    return stiffness(E_eff, I_eff, Le)


def run():
    print('\n=== INTERMEDIATE SIM: Bimetal Strip ===')

    L     = config.L
    n_elem = config.n_elem
    Le    = L / n_elem
    ndof  = 2 * (n_elem + 1)

    # Layer geometry (each half-thickness)
    h_total = config.h
    h1 = h_total / 2   # top layer (Al)
    h2 = h_total / 2   # bottom layer (Steel)
    b  = config.b

    E1 = config.E_layer1   # Al
    E2 = config.E_layer2   # Steel

    # Effective EI for bimetal (transformed section about composite neutral axis)
    # Neutral axis from bottom: y_na = (E1*A1*(h2+h1/2) + E2*A2*(h2/2)) / (E1*A1 + E2*A2)
    A1 = b * h1
    A2 = b * h2
    y_na = (E1 * A1 * (h2 + h1 / 2) + E2 * A2 * (h2 / 2)) / (E1 * A1 + E2 * A2)

    I1 = b * h1 ** 3 / 12 + A1 * (h2 + h1 / 2 - y_na) ** 2
    I2 = b * h2 ** 3 / 12 + A2 * (h2 / 2 - y_na) ** 2

    EI_eff = E1 * I1 + E2 * I2

    # Mismatch curvature (Timoshenko bimetal formula)
    alpha_diff = config.alpha_mismatch   # CTE difference
    delta_T    = config.delta_T
    eps_star   = alpha_diff * delta_T    # mismatch strain

    # κ = 6 * eps_star * (1+m)^2 / (h*(3(1+m)^2 + (1+mn)(m^2 + 1/(mn))))
    # (Simplified: use curvature-load approach)
    # Equivalent bending moment per unit length from mismatch
    M_star = (E1 * A1 * E2 * A2) / (E1 * A1 + E2 * A2) * eps_star * (h1 / 2 + h2 / 2)

    # FEM assembly
    K = np.zeros((ndof, ndof))
    F = np.zeros(ndof)

    for e in range(n_elem):
        ke = stiffness(EI_eff / Le ** 3 * Le ** 3, 1.0 / Le, Le)
        # Override with direct EI
        ke = (EI_eff / Le ** 3) * np.array([
            [12,    6*Le,   -12,    6*Le],
            [6*Le,  4*Le**2, -6*Le,  2*Le**2],
            [-12,  -6*Le,    12,   -6*Le],
            [6*Le,  2*Le**2, -6*Le,  4*Le**2]
        ])
        idx = [2*e, 2*e+1, 2*e+2, 2*e+3]
        K[np.ix_(idx, idx)] += ke

        # Equivalent nodal moments from mismatch strain
        # For uniform κ distribution the consistent equivalent forces are
        # [0, +M_star, 0, -M_star] per element. Adjacent elements cancel at
        # interior nodes, so actual net force = moment boundary terms only.
        # Instead apply: f = [0, M_star, 0, -M_star] which gives correct
        # cantilever integration when summed
        fe = np.array([0.0, -M_star, 0.0, M_star])  # drives beam upward
        F[idx] += fe

    K_ff, F_f, free_dofs, _ = apply_bc(K, F, fixed_dofs=[0, 1])
    u = solve_static(K_ff, F_f, free_dofs, ndof)

    x = np.linspace(0, L, n_elem + 1)
    w = u[0::2]

    plots.plot_bimetal_deflection(x, w, tag='09_bimetal_deflection')
    print(f'  Bimetal max deflection = {w.max()*1e6:.4f} μm')

    # Compare bimetal vs pure electrostrictive at same equivalent voltage
    from simulations.beam_bending import run_cantilever_fem
    _, w_electro, *_ = run_cantilever_fem(config.V_default)
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(x * 1e3, w * 1e6, lw=2.5, color='teal', label='Bimetal (mismatch strain)')
    ax.plot(x * 1e3, w_electro * 1e6, lw=2.5, ls='--', color='crimson',
            label=f'Electrostrictive V={config.V_default}V')
    ax.legend()
    ax.set_xlabel('Position (mm)')
    ax.set_ylabel('Deflection (μm)')
    ax.set_title('Bimetal vs Electrostrictive Actuation Comparison')
    import os; from visualization.plots import SAVE_DIR, _save
    _save(fig, '10_bimetal_vs_electro')
    plt.close(fig)

    print('  [DONE] bimetal simulation')


if __name__ == '__main__':
    run()
