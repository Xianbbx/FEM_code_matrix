"""
visualization/animations.py
Presentation-quality animations saved as GIFs and MP4s.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import cm
import config

ANIM_DIR = config.ANIM_DIR
os.makedirs(ANIM_DIR, exist_ok=True)

STYLE = {
    'figure.facecolor': 'white',
    'axes.facecolor':   '#f0f0f0',
    'axes.grid':        True,
    'grid.alpha':       0.4,
    'font.size':        11,
}
plt.rcParams.update(STYLE)


def _save_gif(anim, name, fps=20):
    path = os.path.join(ANIM_DIR, name + '.gif')
    writer = animation.PillowWriter(fps=fps)
    anim.save(path, writer=writer, dpi=100)
    print(f'  [saved] {path}')
    return path


# ------------------------------------------------------------------ #

def animate_beam_vs_voltage(x, get_w_fn, voltages,
                             tag='beam_voltage_anim'):
    """
    Animate beam deflection as voltage increases.
    get_w_fn(V) -> array of deflections at nodes x.
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    w0 = get_w_fn(voltages[0])
    line, = ax.plot(x * 1e3, w0 * 1e6, color='steelblue', lw=2.5)
    fill = ax.fill_between(x * 1e3, 0, w0 * 1e6, alpha=0.2, color='steelblue')
    ax.set_xlim(0, x[-1] * 1e3)

    all_w = np.array([get_w_fn(V) for V in voltages])
    w_max = np.abs(all_w).max() * 1e6 * 1.2 + 0.01

    ax.set_ylim(-w_max, w_max)
    ax.set_xlabel('Position (mm)')
    ax.set_ylabel('Deflection (μm)')
    title = ax.set_title('')

    def update(i):
        V = voltages[i]
        w = get_w_fn(V)
        line.set_ydata(w * 1e6)
        # re-draw fill
        for coll in ax.collections:
            coll.remove()
        ax.fill_between(x * 1e3, 0, w * 1e6, alpha=0.2, color='steelblue')
        title.set_text(f'Electrostrictive Beam — V = {V:.0f} V  |  '
                       f'Max δ = {w.max()*1e6:.3f} μm')
        return line,

    ani = animation.FuncAnimation(fig, update, frames=len(voltages),
                                  interval=80, blit=False)
    _save_gif(ani, tag)
    plt.close(fig)
    return ani


def animate_sinusoidal_actuation(x, get_w_fn, voltages,
                                  tag='sinusoidal_anim'):
    """
    Animate sinusoidal wave-shape deformation evolution.
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    cmap = cm.cool
    ax.set_xlim(0, x[-1] * 1e3)
    all_w = np.array([get_w_fn(V) for V in voltages])
    ax.set_ylim(all_w.min() * 1e6 * 1.3, all_w.max() * 1e6 * 1.3 + 0.01)
    ax.set_xlabel('Position (mm)')
    ax.set_ylabel('Deflection (μm)')
    line, = ax.plot(x * 1e3, all_w[0] * 1e6, color=cmap(0), lw=2.5)
    title = ax.set_title('')

    def update(i):
        V = voltages[i]
        w = get_w_fn(V)
        line.set_ydata(w * 1e6)
        line.set_color(cmap(i / len(voltages)))
        title.set_text(f'Sinusoidal Actuation — V = {V:.0f} V')
        return line,

    ani = animation.FuncAnimation(fig, update, frames=len(voltages),
                                  interval=80, blit=False)
    _save_gif(ani, tag, fps=15)
    plt.close(fig)
    return ani


def animate_mode_shapes(x, modes, freqs, n_modes=4,
                         tag='mode_shape_anim'):
    """
    Animate each mode oscillating in time.
    """
    T = 60          # frames per mode
    n_show = min(n_modes, modes.shape[1])
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_xlim(0, x[-1] * 1e3)
    ax.set_ylim(-1.2, 1.2)
    ax.set_xlabel('Position (mm)')
    ax.set_ylabel('Normalised Amplitude')
    line, = ax.plot([], [], lw=2.5, color='crimson')
    title = ax.set_title('')

    total_frames = n_show * T

    def update(frame):
        mode_idx = frame // T
        t_phase = (frame % T) / T
        phi = modes[:, mode_idx]
        phi_norm = phi / (np.max(np.abs(phi)) + 1e-30)
        line.set_data(x * 1e3, phi_norm * np.sin(2 * np.pi * t_phase))
        title.set_text(f'Mode {mode_idx+1} — f = {freqs[mode_idx]:.1f} Hz')
        return line,

    ani = animation.FuncAnimation(fig, update, frames=total_frames,
                                  interval=40, blit=False)
    _save_gif(ani, tag, fps=25)
    plt.close(fig)
    return ani


def animate_plastic_zone_growth(x, vm_fn, voltages, sigma_y,
                                 tag='plastic_growth_anim'):
    """
    Show Von Mises stress bar growing and yield zone spreading as V increases.
    vm_fn(V) -> array of σ_vm at node positions x.
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

    all_vm = np.array([vm_fn(V) for V in voltages])
    vm_max = all_vm.max() / 1e6 * 1.2

    ax1.set_xlim(0, x[-1] * 1e3)
    ax1.set_ylim(0, vm_max)
    ax1.set_ylabel('σ_vm (MPa)')
    ax1.axhline(sigma_y / 1e6, color='red', ls='--', lw=1.8, label='σ_y')
    ax1.legend(loc='upper right')

    ax2.set_xlim(0, x[-1] * 1e3)
    ax2.set_ylim(-0.1, 1.5)
    ax2.set_ylabel('Plastic (1=Yes)')
    ax2.set_xlabel('Position (mm)')

    bars1 = ax1.bar(x * 1e3, np.zeros(len(x)),
                    width=(x[-1] - x[0]) / len(x) * 1e3, edgecolor='none', color='steelblue')
    bars2 = ax2.bar(x * 1e3, np.zeros(len(x)),
                    width=(x[-1] - x[0]) / len(x) * 1e3, edgecolor='none', color='green')
    title = ax1.set_title('')

    def update(i):
        V = voltages[i]
        vm = vm_fn(V)
        vm_mpa = vm / 1e6
        plastic = (vm >= sigma_y).astype(float)
        for bar, h in zip(bars1, vm_mpa):
            bar.set_height(h)
            bar.set_color('red' if h >= sigma_y / 1e6 else 'steelblue')
        for bar, p in zip(bars2, plastic):
            bar.set_height(p)
            bar.set_color('red' if p else 'green')
        title.set_text(f'Plastic Zone Growth — V = {V:.0f} V  |  '
                       f'Plastic fraction = {100*plastic.mean():.1f}%')
        return bars1

    ani = animation.FuncAnimation(fig, update, frames=len(voltages),
                                  interval=100, blit=False)
    _save_gif(ani, tag, fps=12)
    plt.close(fig)
    return ani
