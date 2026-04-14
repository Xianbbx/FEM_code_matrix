"""
simulations/boundary_effects.py
Intermediate Simulation: Compare three boundary conditions.
  1. Cantilever (clamped-free)
  2. Clamped-clamped
  3. Simply-supported
"""
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from physics.electrostriction import electro_strain
from fem.assembly import assemble_beam, apply_bc
from fem.solver import solve_static
from visualization import plots


def _solve_bc(fixed_dofs):
    n_elem = config.n_elem
    Le     = config.L / n_elem
    ndof   = 2 * (n_elem + 1)
    eps_e  = electro_strain(config.V_default, config.t, config.M)
    eps_arr= np.full(n_elem, eps_e)

    K, _, F = assemble_beam(n_elem, Le, config.E, config.I,
                             rho=config.rho, A=config.A,
                             eps_e_array=eps_arr)
    K_ff, F_f, free_dofs, _ = apply_bc(K, F, fixed_dofs=fixed_dofs)
    u = solve_static(K_ff, F_f, free_dofs, ndof)
    x = np.linspace(0, config.L, n_elem + 1)
    return x, u[0::2]


def run():
    print('\n=== INTERMEDIATE SIM: Boundary Condition Comparison ===')

    n_nodes = config.n_elem + 1

    bc_configs = {
        'Cantilever (clamped-free)': [0, 1],
        'Clamped-clamped':           [0, 1, 2*(n_nodes-1), 2*(n_nodes-1)+1],
        'Simply-supported':          [0, 2*(n_nodes-1)],   # w only at ends
    }

    results = {}
    for label, fixed in bc_configs.items():
        x, w = _solve_bc(fixed)
        results[label] = (x, w)
        print(f'  {label}: max δ = {w.max()*1e6:.4f} μm')

    plots.plot_boundary_comparison(results, tag='12_bc_comparison')
    print('  [DONE] boundary_effects simulation')


if __name__ == '__main__':
    run()
