"""
simulations/optimization.py
Advanced Simulation: Maximise tip displacement subject to stress constraint.
Uses scipy.optimize.minimize_scalar / brute-force sweep.
Produces:
  - Displacement vs thickness Pareto-style curve
  - Objective function landscape
  - Optimal operating point plot
"""
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from physics.electrostriction import electro_strain
from fem.assembly import assemble_beam, apply_bc
from fem.solver import solve_static, recover_beam_stress
from visualization.plots import _save
import matplotlib.pyplot as plt


def fem_response(V, t):
    """Return (max_disp, max_stress) for given (V, t)."""
    I_t   = config.b * t ** 3 / 12
    A_t   = config.b * t
    n_elem = config.n_elem
    Le    = config.L / n_elem
    ndof  = 2 * (n_elem + 1)

    eps_e = electro_strain(V, t, config.M)
    eps_arr = np.full(n_elem, eps_e)

    K, _, F = assemble_beam(n_elem, Le, config.E, I_t,
                             rho=config.rho, A=A_t,
                             eps_e_array=eps_arr)
    K_ff, F_f, free_dofs, _ = apply_bc(K, F, fixed_dofs=[0, 1])
    u = solve_static(K_ff, F_f, free_dofs, ndof)

    w = u[0::2]
    x_mid, stress, _ = recover_beam_stress(u, n_elem, Le, config.E, I_t, t, eps_arr)
    return w.max(), np.abs(stress).max()


def run():
    print('\n=== ADVANCED SIM: Optimization ===')

    voltages    = np.linspace(20, config.V_max, 20)
    thicknesses = np.linspace(0.3e-3, 2.5e-3, 20)

    VV, TT = np.meshgrid(voltages, thicknesses, indexing='ij')
    DISP   = np.zeros_like(VV)
    STRESS = np.zeros_like(VV)

    for i, V in enumerate(voltages):
        for j, t in enumerate(thicknesses):
            d, s = fem_response(V, t)
            DISP[i, j]   = d
            STRESS[i, j] = s

    # Feasible region: stress < sigma_y
    feasible = STRESS < config.sigma_y

    # Plot 1: displacement landscape
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    c0 = axes[0].contourf(VV, TT * 1e3, DISP * 1e6, levels=20, cmap='hot')
    fig.colorbar(c0, ax=axes[0], label='Max Deflection (μm)')
    axes[0].set_xlabel('Voltage (V)')
    axes[0].set_ylabel('Thickness (mm)')
    axes[0].set_title('Deflection Landscape')

    c1 = axes[1].contourf(VV, TT * 1e3, STRESS / 1e6, levels=20, cmap='RdYlGn_r')
    fig.colorbar(c1, ax=axes[1], label='Max Stress (MPa)')
    axes[1].contour(VV, TT * 1e3, feasible.astype(float),
                    levels=[0.5], colors='blue', linewidths=2)
    axes[1].set_xlabel('Voltage (V)')
    axes[1].set_ylabel('Thickness (mm)')
    axes[1].set_title('Stress Landscape (blue = yield boundary)')

    fig.suptitle('Optimisation Landscape: Deflection vs Stress', fontsize=14)
    _save(fig, '27_optimisation_landscape')
    plt.close(fig)

    # Plot 2: Pareto front (max displacement for each thickness within yield limit)
    pareto_V    = []
    pareto_disp = []
    pareto_t    = []

    for j, t in enumerate(thicknesses):
        best_disp = 0
        best_V    = 0
        for i, V in enumerate(voltages):
            if STRESS[i, j] < config.sigma_y:
                if DISP[i, j] > best_disp:
                    best_disp = DISP[i, j]
                    best_V    = V
        if best_disp > 0:
            pareto_disp.append(best_disp)
            pareto_V.append(best_V)
            pareto_t.append(t)

    fig, ax = plt.subplots(figsize=(9, 5))
    sc = ax.scatter(np.array(pareto_t) * 1e3, np.array(pareto_disp) * 1e6,
                    c=pareto_V, cmap='plasma', s=80, zorder=5)
    ax.plot(np.array(pareto_t) * 1e3, np.array(pareto_disp) * 1e6,
            'k--', lw=1.2, alpha=0.5)
    fig.colorbar(sc, ax=ax, label='Optimal Voltage (V)')
    ax.set_xlabel('Thickness (mm)')
    ax.set_ylabel('Max Feasible Deflection (μm)')
    ax.set_title('Pareto Front: Max Deflection Within Yield Limit')
    _save(fig, '28_pareto_front')
    plt.close(fig)

    print(f'  Global optimum (no yield): disp = {max(pareto_disp)*1e6:.4f} μm')
    print('  [DONE] optimization simulation')


if __name__ == '__main__':
    run()
