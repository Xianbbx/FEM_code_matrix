"""
visualization/plots.py
High-quality, presentation-ready static plots.
All functions save to PLOTS_DIR and optionally show on screen.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D          # noqa: F401
import config

SAVE_DIR = config.PLOTS_DIR
os.makedirs(SAVE_DIR, exist_ok=True)

STYLE = {
    'figure.facecolor': 'white',
    'axes.facecolor':   '#f8f8f8',
    'axes.grid':        True,
    'grid.alpha':       0.4,
    'font.size':        12,
    'axes.titlesize':   14,
    'axes.labelsize':   12,
    'lines.linewidth':  2.0,
}
plt.rcParams.update(STYLE)


def _save(fig, name):
    path = os.path.join(SAVE_DIR, name + '.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    print(f'  [saved] {path}')
    return path


# ------------------------------------------------------------------ #

def plot_beam_deflection(x, w, title='Beam Deflection',
                         bc_label='Cantilever', show=False, tag='beam_deflection'):
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(x * 1e3, w * 1e6, color='steelblue', lw=2.5)
    ax.fill_between(x * 1e3, 0, w * 1e6, alpha=0.15, color='steelblue')
    ax.axhline(0, color='k', lw=0.8, ls='--')
    ax.set_xlabel('Position along beam (mm)')
    ax.set_ylabel('Deflection (μm)')
    ax.set_title(f'{title} — {bc_label}')
    _save(fig, tag)
    if show: plt.show()
    plt.close(fig)


def plot_voltage_vs_displacement(voltages, displacements, show=False, tag='V_vs_disp'):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(voltages ** 2, displacements * 1e6, 'o-', color='crimson', ms=5)
    ax.set_xlabel('V² (V²)')
    ax.set_ylabel('Max Deflection (μm)')
    ax.set_title('Displacement vs V² — Linear Confirms ε ∝ V²')
    _save(fig, tag)
    if show: plt.show()
    plt.close(fig)


def plot_stress_distribution(x_mid, stress, sigma_y=None, show=False, tag='stress_dist'):
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(x_mid * 1e3, stress / 1e6, width=(x_mid[1] - x_mid[0]) * 1e3 * 0.85,
           color='darkorange', alpha=0.8, label='σ_bending (MPa)')
    if sigma_y is not None:
        ax.axhline( sigma_y / 1e6, color='red',  ls='--', lw=1.8, label=f'σ_y = {sigma_y/1e6:.0f} MPa')
        ax.axhline(-sigma_y / 1e6, color='red',  ls='--', lw=1.8)
    ax.set_xlabel('Position (mm)')
    ax.set_ylabel('Bending Stress (MPa)')
    ax.set_title('Bending Stress Distribution Along Beam')
    ax.legend()
    _save(fig, tag)
    if show: plt.show()
    plt.close(fig)


def plot_von_mises_field(x, sigma_vm, sigma_y, show=False, tag='von_mises'):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    # Top: colour-mapped bar
    norm = plt.Normalize(0, max(sigma_vm.max(), sigma_y))
    colours = cm.RdYlGn_r(norm(sigma_vm))
    ax1.bar(x * 1e3, sigma_vm / 1e6,
            width=(x[1] - x[0]) * 1e3,
            color=colours, edgecolor='none')
    ax1.axhline(sigma_y / 1e6, color='k', ls='--', lw=2, label=f'σ_yield = {sigma_y/1e6:.0f} MPa')
    ax1.set_ylabel('Von Mises Stress (MPa)')
    ax1.set_title('Von Mises Stress Field Along Beam')
    ax1.legend()
    sm = cm.ScalarMappable(cmap='RdYlGn_r', norm=norm)
    fig.colorbar(sm, ax=ax1, label='σ_vm (Pa)', fraction=0.03, pad=0.02)

    # Bottom: yield / safe
    plastic = sigma_vm >= sigma_y
    ax2.bar(x * 1e3, np.where(plastic, 1, 0),
            width=(x[1] - x[0]) * 1e3,
            color=np.where(plastic, 'red', 'green'), edgecolor='none')
    ax2.set_ylabel('Yielded (1=Yes)')
    ax2.set_xlabel('Position (mm)')
    ax2.set_title('Plastic Yield Zone')
    _save(fig, tag)
    if show: plt.show()
    plt.close(fig)


def plot_thickness_vs_displacement(thicknesses, displacements, show=False, tag='t_vs_disp'):
    fig, ax = plt.subplots(figsize=(8, 5))
    disp_clipped = np.maximum(np.abs(displacements), 1e-15)
    ax.plot(thicknesses * 1e3, disp_clipped * 1e6, 's-', color='purple')
    ax.set_xlabel('Thickness (mm)')
    ax.set_ylabel('Max Deflection (μm)')
    ax.set_title('Effect of Thickness on Electrostrictive Deflection')
    ax.set_yscale('log')
    _save(fig, tag)
    if show: plt.show()
    plt.close(fig)


def plot_bimetal_deflection(x, w, show=False, tag='bimetal'):
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(x * 1e3, w * 1e6, color='teal', lw=2.5)
    ax.fill_between(x * 1e3, 0, w * 1e6, alpha=0.15, color='teal')
    ax.set_xlabel('Position (mm)')
    ax.set_ylabel('Deflection (μm)')
    ax.set_title('Bimetal Strip Deflection (Mismatch Strain)')
    _save(fig, tag)
    if show: plt.show()
    plt.close(fig)


def plot_sinusoidal_deformation(x, w_list, voltages, show=False, tag='sinusoidal_def'):
    fig, ax = plt.subplots(figsize=(10, 5))
    cmap = cm.cool
    for i, (w, V) in enumerate(zip(w_list, voltages)):
        c = cmap(i / len(voltages))
        ax.plot(x * 1e3, w * 1e6, color=c, lw=1.5, alpha=0.85, label=f'V={V:.0f}')
    sm = cm.ScalarMappable(cmap='cool',
                           norm=plt.Normalize(voltages[0], voltages[-1]))
    fig.colorbar(sm, ax=ax, label='Voltage (V)')
    ax.set_xlabel('Position (mm)')
    ax.set_ylabel('Deflection (μm)')
    ax.set_title('Sinusoidal Actuation — Deformation Evolution')
    _save(fig, tag)
    if show: plt.show()
    plt.close(fig)


def plot_boundary_comparison(results, show=False, tag='bc_comparison'):
    """
    results : dict {label: (x, w)}
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ['steelblue', 'crimson', 'forestgreen']
    for (label, (x, w)), c in zip(results.items(), colors):
        ax.plot(x * 1e3, w * 1e6, lw=2.5, color=c, label=label)
    ax.legend()
    ax.set_xlabel('Position (mm)')
    ax.set_ylabel('Deflection (μm)')
    ax.set_title('Boundary Condition Comparison')
    _save(fig, tag)
    if show: plt.show()
    plt.close(fig)


def plot_mode_shapes(x, modes, freqs, n_modes=6, show=False, tag='mode_shapes'):
    cols = 3
    rows = (n_modes + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(14, rows * 3))
    axes = axes.ravel()
    cmap = cm.tab10
    for i in range(n_modes):
        phi = modes[:, i]
        phi /= (np.max(np.abs(phi)) + 1e-30)
        axes[i].plot(x * 1e3, phi, color=cmap(i / 10), lw=2)
        axes[i].fill_between(x * 1e3, 0, phi, alpha=0.15, color=cmap(i / 10))
        axes[i].axhline(0, color='k', lw=0.7)
        axes[i].set_title(f'Mode {i+1}: {freqs[i]:.1f} Hz', fontsize=10)
        axes[i].set_xlabel('mm', fontsize=8)
        axes[i].tick_params(labelsize=8)
    for j in range(n_modes, len(axes)):
        axes[j].set_visible(False)
    fig.suptitle('Modal Analysis — Mode Shapes', fontsize=15, y=1.01)
    fig.tight_layout()
    _save(fig, tag)
    if show: plt.show()
    plt.close(fig)


def plot_yield_surface_3d(sigma_y, show=False, tag='yield_surface_3d'):
    from physics.plasticity import yield_surface_cylinder
    X, Y, Z = yield_surface_cylinder(sigma_y)
    fig = plt.figure(figsize=(9, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X / 1e6, Y / 1e6, Z / 1e6,
                    alpha=0.45, cmap='plasma', edgecolor='none')
    # Hydrostatic axis
    h_max = 1.5 * sigma_y / np.sqrt(3) / 1e6
    t_a = np.linspace(-h_max, h_max, 50)
    ax.plot(t_a, t_a, t_a, 'k--', lw=1.5, label='Hydrostatic axis')
    ax.set_xlabel('σ₁ (MPa)')
    ax.set_ylabel('σ₂ (MPa)')
    ax.set_zlabel('σ₃ (MPa)')
    ax.set_title('Von Mises Yield Cylinder in Principal Stress Space')
    ax.legend()
    _save(fig, tag)
    if show: plt.show()
    plt.close(fig)


def plot_pi_plane(sigma_y, stress_points=None, show=False, tag='pi_plane'):
    from physics.plasticity import yield_locus_pi_plane
    xi, eta = yield_locus_pi_plane(sigma_y)
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.plot(xi / 1e6, eta / 1e6, 'b-', lw=2.5, label='Von Mises yield locus')
    ax.fill(xi / 1e6, eta / 1e6, alpha=0.08, color='blue')
    ax.axhline(0, color='k', lw=0.6)
    ax.axvline(0, color='k', lw=0.6)
    ax.set_aspect('equal')
    if stress_points is not None:
        for lbl, (x_p, y_p) in stress_points.items():
            ax.plot(x_p / 1e6, y_p / 1e6, 'ro', ms=8)
            ax.annotate(lbl, (x_p / 1e6, y_p / 1e6), textcoords='offset points',
                        xytext=(5, 5), fontsize=9)
    ax.set_xlabel('ξ (MPa)')
    ax.set_ylabel('η (MPa)')
    ax.set_title('π-Plane Projection of Von Mises Yield Criterion')
    ax.legend()
    _save(fig, tag)
    if show: plt.show()
    plt.close(fig)


def plot_plate_deformation(nodes, u_disp, Lx, Ly, show=False, tag='plate_deform'):
    """
    Colour contour of displacement magnitude for 2-D plate.
    nodes: (n_nodes, 2); u_disp: (2*n_nodes,)
    """
    ux = u_disp[0::2]
    uy = u_disp[1::2]
    mag = np.sqrt(ux ** 2 + uy ** 2)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sc0 = axes[0].scatter(nodes[:, 0] * 1e3, nodes[:, 1] * 1e3,
                          c=ux * 1e6, cmap='RdBu', s=30)
    fig.colorbar(sc0, ax=axes[0], label='ux (μm)')
    axes[0].set_title('Plate x-Displacement')
    axes[0].set_xlabel('x (mm)'); axes[0].set_ylabel('y (mm)')
    axes[0].set_aspect('equal')

    sc1 = axes[1].scatter(nodes[:, 0] * 1e3, nodes[:, 1] * 1e3,
                          c=mag * 1e6, cmap='hot', s=30)
    fig.colorbar(sc1, ax=axes[1], label='|u| (μm)')
    axes[1].set_title('Plate Displacement Magnitude')
    axes[1].set_xlabel('x (mm)'); axes[1].set_ylabel('y (mm)')
    axes[1].set_aspect('equal')

    fig.suptitle('2-D Plate FEM — Displacement Field')
    _save(fig, tag)
    if show: plt.show()
    plt.close(fig)


def plot_plate_stress(nodes, sigma_vm, sigma_y, show=False, tag='plate_stress'):
    fig, ax = plt.subplots(figsize=(10, 6))
    sc = ax.scatter(nodes[:, 0] * 1e3, nodes[:, 1] * 1e3,
                    c=sigma_vm / 1e6, cmap='jet', s=40,
                    vmin=0, vmax=sigma_vm.max() / 1e6)
    fig.colorbar(sc, ax=ax, label='σ_vm (MPa)')
    yield_mask = sigma_vm >= sigma_y
    if yield_mask.any():
        ax.scatter(nodes[yield_mask, 0] * 1e3, nodes[yield_mask, 1] * 1e3,
                   marker='x', s=60, color='red', label='Yielded')
        ax.legend()
    ax.set_aspect('equal')
    ax.set_xlabel('x (mm)'); ax.set_ylabel('y (mm)')
    ax.set_title('Von Mises Stress Field — 2-D Plate')
    _save(fig, tag)
    if show: plt.show()
    plt.close(fig)


def plot_energy_vs_voltage(voltages, energies, show=False, tag='energy_vs_V'):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(voltages, np.array(energies) * 1e9, 'o-', color='darkorchid', ms=5)
    ax.set_xlabel('Voltage (V)')
    ax.set_ylabel('Stored Energy (nJ)')
    ax.set_title('Electrostrictive Energy vs Applied Voltage')
    _save(fig, tag)
    if show: plt.show()
    plt.close(fig)


def plot_mohr_circle(sxx, syy, sxy, sigma_y=None, show=False, tag='mohr_circle'):
    from utils.stress_utils import stress_mohr_circle_2d, mohr_circle_points
    center, radius, angle = stress_mohr_circle_2d(sxx, syy, sxy)
    s_n, tau = mohr_circle_points(center, radius)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(s_n / 1e6, tau / 1e6, 'b-', lw=2)
    ax.plot([sxx / 1e6, syy / 1e6], [sxy / 1e6, -sxy / 1e6], 'ko', ms=8)
    ax.axhline(0, color='k', lw=0.6)
    ax.axvline(0, color='k', lw=0.6)
    ax.set_aspect('equal')
    if sigma_y is not None:
        ax.axvline( sigma_y / 1e6, color='r', ls='--', lw=1.5, label='±σ_y')
        ax.axvline(-sigma_y / 1e6, color='r', ls='--', lw=1.5)
        ax.legend()
    ax.set_xlabel('Normal Stress (MPa)')
    ax.set_ylabel('Shear Stress (MPa)')
    ax.set_title("Mohr's Circle")
    _save(fig, tag)
    if show: plt.show()
    plt.close(fig)


def plot_distortion_energy(x, s1_arr, s2_arr, sigma_y, show=False, tag='distortion_energy'):
    from physics.plasticity import distortion_energy
    phi = distortion_energy(s1_arr, s2_arr, np.zeros_like(s1_arr))
    phi_yield = distortion_energy(sigma_y, 0.0, 0.0)

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(x * 1e3, phi / phi_yield, color='saddlebrown', lw=2)
    ax.axhline(1.0, color='red', ls='--', lw=1.8, label='Yield threshold φ = 1')
    ax.fill_between(x * 1e3, phi / phi_yield, 1,
                    where=(phi / phi_yield) >= 1,
                    alpha=0.5, color='red', label='Plastic zone')
    ax.set_xlabel('Position (mm)')
    ax.set_ylabel('φ_distortion / φ_yield')
    ax.set_title('Distortion Energy — Yield Trigger Visualisation')
    ax.legend()
    _save(fig, tag)
    if show: plt.show()
    plt.close(fig)


def plot_volumetric_vs_deviatoric(s1_arr, s2_arr, s3_arr=None, show=False, tag='vol_dev'):
    if s3_arr is None:
        s3_arr = np.zeros_like(s1_arr)
    sv = (s1_arr + s2_arr + s3_arr) / 3.0
    dev_norm = np.sqrt(
        (s1_arr - sv) ** 2 + (s2_arr - sv) ** 2 + (s3_arr - sv) ** 2
    )

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(np.arange(len(sv)), sv / 1e6, label='Volumetric σ_vol (MPa)', color='navy')
    ax.plot(np.arange(len(sv)), dev_norm / 1e6, label='Deviatoric |s_dev| (MPa)', color='darkorange')
    ax.axhline(0, color='k', lw=0.5)
    ax.set_xlabel('Element index')
    ax.set_ylabel('Stress (MPa)')
    ax.set_title('Volumetric vs Deviatoric Stress — Only Deviatoric Causes Yielding')
    ax.legend()
    _save(fig, tag)
    if show: plt.show()
    plt.close(fig)


def plot_plastic_zone_growth(x, vm_per_voltage, voltages, sigma_y,
                             show=False, tag='plastic_zone_growth'):
    fig, ax = plt.subplots(figsize=(10, 6))
    cmap = cm.hot_r
    for i, (vm, V) in enumerate(zip(vm_per_voltage, voltages)):
        plastic_fraction = np.sum(vm >= sigma_y) / len(vm)
        c = cmap(i / len(voltages))
        ax.bar(V, plastic_fraction * 100, color=c, width=(voltages[1] - voltages[0]) * 0.8)
    sm = cm.ScalarMappable(cmap='hot_r', norm=plt.Normalize(voltages[0], voltages[-1]))
    fig.colorbar(sm, ax=ax, label='Voltage (V)')
    ax.set_xlabel('Voltage (V)')
    ax.set_ylabel('Plastic Volume Fraction (%)')
    ax.set_title('Plastic Zone Growth with Increasing Voltage')
    _save(fig, tag)
    if show: plt.show()
    plt.close(fig)
