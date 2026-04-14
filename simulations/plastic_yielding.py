"""
simulations/plastic_yielding.py
Advanced Simulation: Plasticity analysis of electrostrictive beam.
Produces:
  - Von Mises heatmap
  - Plastic zone growth animation
  - Yield surface (3-D cylinder)
  - π-plane locus
  - Volumetric vs deviatoric plot
  - Distortion energy visualisation
  - Stress state evolution with voltage
"""
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from physics.electrostriction import electro_strain
from physics.plasticity import (von_mises_from_principals, check_yield,
                                  distortion_energy, volumetric_deviatoric_split)
from fem.assembly import assemble_beam, apply_bc
from fem.solver import solve_static, recover_beam_stress
from visualization import plots, animations
import matplotlib.pyplot as plt
from visualization.plots import _save


def compute_vm_at_voltage(V):
    """Run FEM and return Von Mises stresses at element midpoints."""
    n_elem = config.n_elem
    Le     = config.L / n_elem
    ndof   = 2 * (n_elem + 1)
    eps_e  = electro_strain(V, config.t, config.M)
    eps_arr = np.full(n_elem, eps_e)

    K, _, F = assemble_beam(n_elem, Le, config.E, config.I,
                             rho=config.rho, A=config.A,
                             eps_e_array=eps_arr)
    K_ff, F_f, free_dofs, _ = apply_bc(K, F, fixed_dofs=[0, 1])
    u = solve_static(K_ff, F_f, free_dofs, ndof)

    x_mid, stress, _ = recover_beam_stress(u, n_elem, Le, config.E, config.I,
                                           config.h, eps_arr)
    # 1-D beam bending: σ_1=stress, σ_2=0, σ_3=0
    sigma_vm = np.abs(stress)   # equivalent to von_mises_from_principals(s,0,0)
    return x_mid, sigma_vm, stress


def run():
    print('\n=== ADVANCED SIM: Plastic Yielding ===')

    sigma_y = config.sigma_y

    # 1. Von Mises at high voltage
    V_high = config.V_max
    x_mid, vm_high, stress_high = compute_vm_at_voltage(V_high)
    plots.plot_von_mises_field(x_mid, vm_high, sigma_y,
                               tag='15_von_mises_high_V')

    # 2. Stress distribution at high V
    plots.plot_stress_distribution(x_mid, stress_high, sigma_y=sigma_y,
                                   tag='16_stress_high_V')

    # 3. Yield surface 3-D
    plots.plot_yield_surface_3d(sigma_y, tag='17_yield_surface_3d')

    # 4. π-plane
    # Project current stress state: s1=σ, s2=0, s3=0
    # π-plane coords: ξ = (s1-s2)/√2, η = (s1+s2-2*s3)/√6
    s1 = stress_high
    xi_pts  = s1 / np.sqrt(2)
    eta_pts = s1 / np.sqrt(6)
    stress_pts = {f'e{i}': (xi_pts[i], eta_pts[i])
                  for i in np.linspace(0, len(s1)-1, 6, dtype=int)}
    plots.plot_pi_plane(sigma_y, stress_points=stress_pts, tag='18_pi_plane')

    # 5. Volumetric vs deviatoric
    s2 = np.zeros_like(s1)
    s3 = np.zeros_like(s1)
    plots.plot_volumetric_vs_deviatoric(s1, s2, s3,
                                         tag='19_vol_vs_dev')

    # 6. Distortion energy
    plots.plot_distortion_energy(x_mid, s1, s2, sigma_y,
                                  tag='20_distortion_energy')

    # 7. Plastic zone growth
    voltages = np.linspace(50, config.V_max, 40)
    vm_per_V = []
    for V in voltages:
        xm, vm, _= compute_vm_at_voltage(V)
        vm_per_V.append(vm)
    plots.plot_plastic_zone_growth(x_mid, vm_per_V, voltages, sigma_y,
                                    tag='21_plastic_zone_growth')

    # 8. Plastic growth animation
    def vm_fn(V):
        xm, vm, _ = compute_vm_at_voltage(V)
        return vm
    animations.animate_plastic_zone_growth(x_mid, vm_fn, voltages, sigma_y,
                                            tag='04_plastic_growth_anim')

    # 9. Stress evolution surface (V × position)
    VV = np.linspace(0, config.V_max, 50)
    all_stress = []
    for V in VV:
        _, _, s = compute_vm_at_voltage(V)
        all_stress.append(s)
    all_stress = np.array(all_stress)   # (n_V, n_elem)

    from mpl_toolkits.mplot3d import Axes3D   # noqa
    XV, XX = np.meshgrid(VV, x_mid * 1e3, indexing='ij')
    fig = plt.figure(figsize=(12, 7))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(XV, XX, all_stress / 1e6, cmap='viridis', edgecolor='none', alpha=0.85)
    ax.set_xlabel('Voltage (V)')
    ax.set_ylabel('Position (mm)')
    ax.set_zlabel('Bending Stress (MPa)')
    ax.set_title('Stress Evolution — Voltage × Position Surface')
    _save(fig, '22_stress_surface')
    plt.close(fig)

    # 10. Hydrostatic stress does NOT cause yielding —explicit demonstration
    sv_arr = [(s1[i] + s2[i] + s3[i]) / 3.0 for i in range(len(s1))]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(x_mid * 1e3, np.array(sv_arr) / 1e6, 'navy', lw=2,
            label='σ_hydrostatic (MPa)')
    ax.plot(x_mid * 1e3, vm_high / 1e6, 'crimson', lw=2,
            label='σ_vm (MPa)')
    ax.axhline(sigma_y / 1e6, color='k', ls='--', lw=1.5, label='σ_y')
    ax.legend()
    ax.set_xlabel('Position (mm)')
    ax.set_ylabel('Stress (MPa)')
    ax.set_title('Hydrostatic Stress vs Von Mises:\nOnly Deviatoric Triggers Yielding')
    _save(fig, '23_hydrostatic_vs_vm')
    plt.close(fig)

    print('  [DONE] plastic_yielding simulation')


if __name__ == '__main__':
    run()
