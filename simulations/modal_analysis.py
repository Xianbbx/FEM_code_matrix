"""
simulations/modal_analysis.py
Advanced Simulation: Modal analysis of electrostrictive beam.
Solves generalised eigenvalue problem K φ = λ M φ.
Produces:
  - Natural frequency table
  - Mode shape subplot
  - Mode shape animation
"""
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from fem.assembly import assemble_beam, apply_bc
from fem.solver import solve_modal
from visualization import plots, animations


def run():
    print('\n=== ADVANCED SIM: Modal Analysis ===')

    n_elem = config.n_elem
    Le     = config.L / n_elem
    ndof   = 2 * (n_elem + 1)

    K, Mg, _ = assemble_beam(n_elem, Le, config.E, config.I,
                              rho=config.rho, A=config.A)
    K_ff, _, free_dofs, M_ff = apply_bc(K, np.zeros(ndof), [0, 1], M=Mg)

    n_modes = config.n_modes
    freqs, mode_vecs = solve_modal(K_ff, M_ff, n_modes=n_modes)

    # Reconstruct full mode shapes
    x         = np.linspace(0, config.L, n_elem + 1)
    n_free    = len(free_dofs)
    modes_full = np.zeros((n_elem + 1, n_modes))

    for i in range(n_modes):
        u_full = np.zeros(ndof)
        u_full[free_dofs] = mode_vecs[:, i]
        modes_full[:, i] = u_full[0::2]  # transverse DOFs

    print('  Natural frequencies (Hz):')
    for i, f in enumerate(freqs):
        print(f'    Mode {i+1}: {f:.2f} Hz')

    plots.plot_mode_shapes(x, modes_full, freqs, n_modes=n_modes,
                            tag='13_mode_shapes')

    animations.animate_mode_shapes(x, modes_full, freqs, n_modes=min(4, n_modes),
                                    tag='03_mode_shapes_anim')

    # Frequency bar chart
    import matplotlib.pyplot as plt
    from visualization.plots import _save
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(range(1, n_modes + 1), freqs, color='steelblue', edgecolor='navy')
    ax.set_xlabel('Mode Number')
    ax.set_ylabel('Frequency (Hz)')
    ax.set_title('Natural Frequencies of Electrostrictive Beam')
    for i, f in enumerate(freqs):
        ax.text(i + 1, f + freqs[-1] * 0.01, f'{f:.1f}', ha='center', fontsize=9)
    _save(fig, '14_freq_bar')
    plt.close(fig)

    print('  [DONE] modal_analysis simulation')


if __name__ == '__main__':
    run()
