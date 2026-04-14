"""
simulations/beam_bending.py
Basic Simulation 1: Cantilever beam electrostrictive actuation.
Produces:
  - Beam deflection profile
  - Voltage vs displacement (quadratic validation)
  - Thickness sensitivity
  - Stress distribution + Von Mises
  - Energy vs voltage
  - Mohr's circle at max-stress point
  - Deformation animation
"""
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from physics.electrostriction import electro_strain
from fem.assembly import assemble_beam, apply_bc
from fem.solver import solve_static, recover_beam_stress
from physics.plasticity import von_mises_from_principals
from visualization import plots, animations


def run_cantilever_fem(V, E=None, I=None, L=None, n_elem=None,
                       rho=None, A=None):
    """Core FEM solve for cantilever beam under electrostrictive load."""
    E_     = E      or config.E
    I_     = I      or config.I
    L_     = L      or config.L
    n_elem_= n_elem or config.n_elem
    rho_   = rho    or config.rho
    A_     = A      or config.A
    Le     = L_ / n_elem_
    ndof   = 2 * (n_elem_ + 1)

    eps_e  = electro_strain(V, config.t, config.M)
    eps_arr = np.full(n_elem_, eps_e)

    K, Mg, F = assemble_beam(n_elem_, Le, E_, I_, rho=rho_, A=A_,
                              eps_e_array=eps_arr)
    K_ff, F_f, free_dofs, _ = apply_bc(K, F, fixed_dofs=[0, 1])
    u = solve_static(K_ff, F_f, free_dofs, ndof)

    x = np.linspace(0, L_, n_elem_ + 1)
    w = u[0::2]
    return x, w, u, n_elem_, Le, E_, I_


def run():
    print('\n=== BASIC SIM 1: Cantilever Beam Bending ===')

    # 1. Single voltage — deflection profile
    V = config.V_default
    x, w, u, n_elem, Le, E, I = run_cantilever_fem(V)
    plots.plot_beam_deflection(x, w, tag='01_beam_deflection',
                               title=f'Cantilever  V={V:.0f}V')
    print(f'  Max deflection = {w.max()*1e6:.4f} μm  @ V={V} V')

    # 2. Voltage sweep — validate ε ∝ V²
    voltages = np.linspace(0, config.V_max, 50)
    max_disps = []
    for Vs in voltages:
        _, ws, *_ = run_cantilever_fem(Vs)
        max_disps.append(ws.max())
    plots.plot_voltage_vs_displacement(voltages, np.array(max_disps),
                                       tag='02_voltage_vs_disp')

    # 3. Thickness sensitivity
    thicknesses = np.linspace(0.5e-3, 3e-3, 30)
    t_disps = []
    for t in thicknesses:
        I_t = config.b * t ** 3 / 12
        A_t = config.b * t
        eps_e = config.M * (V / t) ** 2
        n_e = config.n_elem
        Le_t = config.L / n_e
        ndof_t = 2 * (n_e + 1)
        eps_arr = np.full(n_e, eps_e)
        from fem.assembly import assemble_beam, apply_bc
        K, _, F = assemble_beam(n_e, Le_t, config.E, I_t,
                                rho=config.rho, A=A_t,
                                eps_e_array=eps_arr)
        K_ff, F_f, free_dofs, _ = apply_bc(K, F, fixed_dofs=[0, 1])
        u_t = solve_static(K_ff, F_f, free_dofs, ndof_t)
        t_disps.append(u_t[0::2].max())
    plots.plot_thickness_vs_displacement(thicknesses, np.array(t_disps),
                                         tag='03_thickness_vs_disp')

    # 4. Stress distribution
    eps_arr = np.full(n_elem, electro_strain(V, config.t, config.M))
    x_mid, stress, kappa = recover_beam_stress(u, n_elem, Le, E, I,
                                               config.h, eps_arr)
    plots.plot_stress_distribution(x_mid, stress, sigma_y=config.sigma_y,
                                   tag='04_stress_distribution')

    # 5. Von Mises field  (1-D beam: σ_vm ≈ |σ_bending|)
    s1 = stress
    sigma_vm = np.abs(s1)
    plots.plot_von_mises_field(x_mid, sigma_vm, config.sigma_y,
                               tag='05_von_mises_beam')

    # 6. Energy vs voltage
    from physics.electrostriction import electrostrictive_energy
    volume = config.A * config.L
    energies = [electrostrictive_energy(Vs, config.t, config.M, volume)
                for Vs in voltages]
    plots.plot_energy_vs_voltage(voltages, energies, tag='06_energy_vs_V')

    # 7. Mohr's circle at peak-stress element
    peak_idx = np.argmax(np.abs(stress))
    sxx = stress[peak_idx]
    plots.plot_mohr_circle(sxx, 0.0, 0.0, sigma_y=config.sigma_y,
                           tag='07_mohr_circle')

    # 8. Distortion energy
    plots.plot_distortion_energy(x_mid, stress, np.zeros_like(stress),
                                 config.sigma_y, tag='08_distortion_energy')

    # 9. Animation: beam bending vs voltage
    v_anim = np.linspace(0, config.V_max, 60)
    def get_w(Vs):
        _, ws, *_ = run_cantilever_fem(Vs)
        return ws
    animations.animate_beam_vs_voltage(x, get_w, v_anim,
                                        tag='01_beam_voltage_anim')

    print('  [DONE] beam_bending simulation')


if __name__ == '__main__':
    run()
