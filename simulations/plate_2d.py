"""
simulations/plate_2d.py
Advanced Simulation: 2-D plate FEM with CST elements.
Electrostrictive body force applied in x-direction.
Produces:
  - Displacement field colour maps
  - Von Mises stress field
  - Deformed mesh
"""
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from physics.elasticity import D_matrix_2D_plane_stress
from physics.plasticity import von_mises_from_components
from physics.electrostriction import electro_strain
from fem.plate_element import mesh_rect_cst, assemble_plate, cst_stress
from visualization import plots
import matplotlib.pyplot as plt
from visualization.plots import _save


def run():
    print('\n=== ADVANCED SIM: 2-D Plate FEM ===')

    Lx, Ly = config.Lx, config.Ly
    nx, ny  = config.nx, config.ny
    t_plate = config.t

    D = D_matrix_2D_plane_stress(config.E, config.nu)

    nodes, elements = mesh_rect_cst(Lx, Ly, nx, ny)
    n_nodes = nodes.shape[0]
    ndof    = 2 * n_nodes

    K, Bs, As = assemble_plate(nodes, elements, D, thickness=t_plate)

    # Electrostrictive body force: uniform ε_xx = M (V/t)² applied as an axial load
    eps_e = electro_strain(config.V_default, config.t, config.M)
    F = np.zeros(ndof)

    # Convert eigen-strain to equivalent nodal forces
    # F_eq = ∫ B^T D ε_eigen dV  → summed over elements
    eps_eigen = np.array([eps_e, 0.0, 0.0])   # only εxx
    for ie, elem in enumerate(elements):
        B = Bs[ie]
        if B is None:
            continue
        A_e   = As[ie]
        f_eq  = B.T @ D @ eps_eigen * A_e * t_plate  # (6,)
        dofs  = []
        for nd in elem:
            dofs += [2*nd, 2*nd+1]
        F[dofs] += f_eq

    # Boundary conditions: left edge clamped (x=0)
    left_nodes = np.where(nodes[:, 0] < 1e-9)[0]
    fixed_dofs = []
    for n in left_nodes:
        fixed_dofs += [2*n, 2*n+1]

    all_dofs  = np.arange(ndof)
    free_dofs = np.setdiff1d(all_dofs, fixed_dofs)

    K_ff = K[np.ix_(free_dofs, free_dofs)]
    F_f  = F[free_dofs]

    u_free = np.linalg.solve(K_ff, F_f)
    u = np.zeros(ndof)
    u[free_dofs] = u_free

    plots.plot_plate_deformation(nodes, u, Lx, Ly, tag='24_plate_displacement')

    # --- Stress recovery at element centroids ---
    node_sigma_vm = np.zeros(n_nodes)
    node_count    = np.zeros(n_nodes)

    for ie, elem in enumerate(elements):
        B = Bs[ie]
        if B is None:
            continue
        dofs = []
        for nd in elem:
            dofs += [2*nd, 2*nd+1]
        u_e = u[dofs]
        sigma, _ = cst_stress(u_e, B, D, eps_eigen=eps_eigen)
        sxx, syy, sxy = sigma
        vm = von_mises_from_components(sxx, syy, sxy=sxy)
        for nd in elem:
            node_sigma_vm[nd] += vm
            node_count[nd]    += 1

    node_count = np.maximum(node_count, 1)
    node_sigma_vm /= node_count

    plots.plot_plate_stress(nodes, node_sigma_vm, config.sigma_y,
                             tag='25_plate_stress')

    # Deformed mesh overlay
    ux = u[0::2]
    uy = u[1::2]
    scale = 1e4   # exaggerate deformation for visualisation
    nodes_def = nodes + scale * np.column_stack([ux, uy])

    fig, ax = plt.subplots(figsize=(12, 6))
    for elem in elements:
        pts_orig = nodes[np.append(elem, elem[0])]
        pts_def  = nodes_def[np.append(elem, elem[0])]
        ax.plot(pts_orig[:, 0] * 1e3, pts_orig[:, 1] * 1e3,
                'b-', lw=0.4, alpha=0.5)
        ax.plot(pts_def[:, 0] * 1e3, pts_def[:, 1] * 1e3,
                'r-', lw=0.6, alpha=0.7)
    import matplotlib.patches as mpatches
    ax.legend(handles=[
        mpatches.Patch(color='blue', label='Undeformed'),
        mpatches.Patch(color='red',  label=f'Deformed (×{scale:.0f})')
    ])
    ax.set_xlabel('x (mm)'); ax.set_ylabel('y (mm)')
    ax.set_aspect('equal')
    ax.set_title('2-D Plate FEM — Deformed vs Undeformed Mesh')
    _save(fig, '26_plate_mesh_deformed')
    plt.close(fig)

    print(f'  Max |u| = {np.sqrt(ux**2 + uy**2).max()*1e6:.4f} μm')
    print('  [DONE] plate_2d simulation')


if __name__ == '__main__':
    run()
