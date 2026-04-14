"""
gui.py — Streamlit interactive GUI for the Electrostrictive FEM Engine
Run: streamlit run gui.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

import config
from physics.electrostriction import electro_strain, spatial_electro_strain
from physics.plasticity import (von_mises_from_principals, yield_locus_pi_plane,
                                  distortion_energy, volumetric_deviatoric_split)
from physics.elasticity import D_matrix_2D_plane_stress
from fem.assembly import assemble_beam, apply_bc
from fem.solver import solve_static, solve_modal, recover_beam_stress
from fem.plate_element import mesh_rect_cst, assemble_plate, cst_stress

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Electrostrictive FEM Lab",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .metric-box {background:#1e1e2e;border-radius:8px;padding:12px 16px;margin:4px 0}
    .stTabs [data-baseweb="tab"] {font-size:15px; font-weight:600}
    h1 {color:#a6e3a1}
    h2 {color:#89b4fa}
    h3 {color:#cba6f7}
</style>
""", unsafe_allow_html=True)

st.title("⚡ Electrostrictive FEM + Plasticity Simulation Engine")
st.caption("IIT Bombay AE246 Structures Lab | Interactive Analysis Platform")

# ─────────────────────────────────────────────
# SIDEBAR — GLOBAL PARAMETERS
# ─────────────────────────────────────────────
st.sidebar.header("⚙️ Material & Geometry")

E_gpa    = st.sidebar.slider("Young's Modulus E (GPa)", 10.0, 300.0, 70.0, 5.0)
sigma_y_ = st.sidebar.slider("Yield Stress σ_y (MPa)", 50.0, 800.0, 250.0, 10.0)
nu_      = st.sidebar.slider("Poisson's Ratio ν", 0.1, 0.49, 0.33, 0.01)
L_mm     = st.sidebar.slider("Beam Length L (mm)", 20.0, 300.0, 100.0, 5.0)
t_mm     = st.sidebar.slider("Thickness t (mm)", 0.2, 5.0, 1.0, 0.1)
b_mm     = st.sidebar.slider("Width b (mm)", 1.0, 20.0, 5.0, 0.5)
M_exp    = st.sidebar.slider("Electrostrictive coeff M (×10⁻¹⁸ m²/V²)", 0.1, 50.0, 1.0, 0.1)
n_elem_  = st.sidebar.slider("FEM Elements", 10, 80, 40, 5)

st.sidebar.divider()
st.sidebar.header("⚡ Voltage Control")
V_       = st.sidebar.slider("Applied Voltage V (V)", 0.0, 500.0, 100.0, 5.0)
V_max_   = st.sidebar.slider("Sweep Max Voltage (V)", 100.0, 1000.0, 300.0, 50.0)

# Derived
E_       = E_gpa * 1e9
sigma_y  = sigma_y_ * 1e6
L_       = L_mm * 1e-3
t_       = t_mm * 1e-3
b_       = b_mm * 1e-3
M_       = M_exp * 1e-18
I_       = b_ * t_**3 / 12
A_       = b_ * t_
rho_     = config.rho
n_elem   = n_elem_
Le_      = L_ / n_elem
ndof_    = 2 * (n_elem + 1)

# ─────────────────────────────────────────────
# CORE FEM HELPERS
# ─────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def fem_cantilever(V, E, I, L, t, b, M_coeff, n_elem, rho):
    Le = L / n_elem
    ndof = 2 * (n_elem + 1)
    eps_e = M_coeff * (V / t) ** 2
    eps_arr = np.full(n_elem, eps_e)
    K, Mg, F = assemble_beam(n_elem, Le, E, I, rho=rho, A=b*t,
                              eps_e_array=eps_arr)
    K_ff, F_f, free_dofs, M_ff = apply_bc(K, F, [0, 1], M=Mg)
    u = solve_static(K_ff, F_f, free_dofs, ndof)
    x = np.linspace(0, L, n_elem + 1)
    w = u[0::2]
    x_mid, stress, kappa = recover_beam_stress(u, n_elem, Le, E, I, t, eps_arr)
    return x, w, x_mid, stress, kappa, eps_e, K_ff, M_ff, free_dofs, ndof, u


@st.cache_data(show_spinner=False)
def fem_modal(E, I, L, t, b, rho, n_elem, n_modes=6):
    Le = L / n_elem
    ndof = 2 * (n_elem + 1)
    K, Mg, _ = assemble_beam(n_elem, Le, E, I, rho=rho, A=b*t)
    K_ff, _, free_dofs, M_ff = apply_bc(K, np.zeros(ndof), [0, 1], M=Mg)
    freqs, mode_vecs = solve_modal(K_ff, M_ff, n_modes=n_modes)
    modes_full = np.zeros((n_elem + 1, n_modes))
    for i in range(n_modes):
        u_full = np.zeros(ndof)
        u_full[free_dofs] = mode_vecs[:, i]
        modes_full[:, i] = u_full[0::2]
    return freqs, modes_full

# ─────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────
tabs = st.tabs([
    "🏗️ Beam Analysis",
    "📈 Voltage Sweep",
    "🎵 Modal Analysis",
    "🔴 Plasticity",
    "🔬 Resonance Explorer",
    "🎯 Multi-Patch Actuator",
    "🧠 Inverse Problem",
    "⚖️ Design Optimiser",
    "🌐 2D Plate",
])

# ══════════════════════════════════════════════
# TAB 1 — BEAM ANALYSIS
# ══════════════════════════════════════════════
with tabs[0]:
    st.header("Cantilever Beam — Electrostrictive Actuation")

    with st.spinner("Solving FEM..."):
        x, w, x_mid, stress, kappa, eps_e, K_ff, M_ff, free_dofs, ndof, u = fem_cantilever(
            V_, E_, I_, L_, t_, b_, M_, n_elem, rho_)

    # Metrics row
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Tip Deflection", f"{w.max()*1e6:.4f} μm")
    c2.metric("Electrostrictive Strain", f"{eps_e:.3e}")
    c3.metric("Peak Bending Stress", f"{np.abs(stress).max()/1e6:.2f} MPa")
    c4.metric("Yielded Elements",
              f"{int(np.sum(np.abs(stress) >= sigma_y))}/{n_elem}",
              delta="⚠️ Plastic!" if np.any(np.abs(stress) >= sigma_y) else "✅ Elastic")

    col1, col2 = st.columns(2)

    # Deflection plot
    with col1:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=x*1e3, y=w*1e6,
            mode='lines', name='Deflection',
            line=dict(color='#89b4fa', width=3),
            fill='tozeroy', fillcolor='rgba(137,180,250,0.15)'
        ))
        fig.add_hline(y=0, line_dash='dash', line_color='white', line_width=1)
        fig.update_layout(
            title=f"Deflection Profile  (V = {V_:.0f} V)",
            xaxis_title="Position (mm)", yaxis_title="Deflection (μm)",
            template='plotly_dark', height=350
        )
        st.plotly_chart(fig, use_container_width=True)

    # Stress + yield
    with col2:
        colours = ['#f38ba8' if abs(s) >= sigma_y else '#a6e3a1' for s in stress]
        fig2 = go.Figure()
        fig2.add_trace(go.Bar(
            x=x_mid*1e3, y=stress/1e6,
            marker_color=colours, name='Bending Stress'
        ))
        fig2.add_hline(y= sigma_y/1e6, line_dash='dot', line_color='#f38ba8',
                       annotation_text=f"σ_y = {sigma_y_:.0f} MPa")
        fig2.add_hline(y=-sigma_y/1e6, line_dash='dot', line_color='#f38ba8')
        fig2.update_layout(
            title="Bending Stress Distribution  (red = yielded)",
            xaxis_title="Position (mm)", yaxis_title="Stress (MPa)",
            template='plotly_dark', height=350
        )
        st.plotly_chart(fig2, use_container_width=True)

    # Curvature + stress through thickness
    col3, col4 = st.columns(2)
    with col3:
        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(
            x=x_mid*1e3, y=kappa,
            mode='lines+markers', marker=dict(size=4),
            line=dict(color='#cba6f7', width=2)
        ))
        fig3.update_layout(
            title="Curvature κ Along Beam",
            xaxis_title="Position (mm)", yaxis_title="Curvature (1/m)",
            template='plotly_dark', height=300
        )
        st.plotly_chart(fig3, use_container_width=True)

    with col4:
        # Stress through thickness at peak element
        peak_e = np.argmax(np.abs(stress))
        z_norm = np.linspace(-1, 1, 100)
        z_phys = z_norm * t_ / 2
        s_top  = stress[peak_e]   # top fibre
        s_through = s_top * z_norm   # linear
        fig4 = go.Figure()
        fig4.add_trace(go.Scatter(
            x=s_through/1e6, y=z_phys*1e3,
            mode='lines', line=dict(color='#fab387', width=3)
        ))
        fig4.add_vline(x= sigma_y/1e6, line_dash='dot', line_color='#f38ba8')
        fig4.add_vline(x=-sigma_y/1e6, line_dash='dot', line_color='#f38ba8')
        fig4.update_layout(
            title=f"Stress Through Thickness (element {peak_e+1})",
            xaxis_title="σ_xx (MPa)", yaxis_title="z (mm)",
            template='plotly_dark', height=300
        )
        st.plotly_chart(fig4, use_container_width=True)


# ══════════════════════════════════════════════
# TAB 2 — VOLTAGE SWEEP
# ══════════════════════════════════════════════
with tabs[1]:
    st.header("Voltage Sweep Analysis")

    n_sweep = st.slider("Number of sweep points", 10, 100, 40, key='vsweep')
    voltages = np.linspace(0, V_max_, n_sweep)

    with st.spinner("Sweeping voltages..."):
        disps, peak_stresses, energies = [], [], []
        for Vs in voltages:
            _, ww, _, ss, _, ee, *_ = fem_cantilever(
                Vs, E_, I_, L_, t_, b_, M_, n_elem, rho_)
            disps.append(ww.max())
            peak_stresses.append(np.abs(ss).max())
            volume = A_ * L_
            energies.append(0.5 * E_ * ee**2 * volume)

    disps = np.array(disps)
    peak_stresses = np.array(peak_stresses)

    col1, col2 = st.columns(2)

    with col1:
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Scatter(
            x=voltages**2, y=disps*1e6,
            mode='lines+markers', name='Tip Deflection (μm)',
            line=dict(color='#89b4fa', width=2.5), marker=dict(size=5)
        ), secondary_y=False)
        # Theoretical linear fit
        if voltages.max() > 0:
            slope = disps[-1] / (voltages[-1]**2 + 1e-30)
            fig.add_trace(go.Scatter(
                x=voltages**2, y=slope*voltages**2*1e6,
                mode='lines', name='Linear fit (ε∝V²)',
                line=dict(color='#a6e3a1', width=1.5, dash='dash')
            ), secondary_y=False)
        fig.update_layout(
            title="Deflection vs V² — Validates Electrostrictive Physics",
            xaxis_title="V² (V²)", template='plotly_dark', height=380
        )
        fig.update_yaxes(title_text="Tip Deflection (μm)", secondary_y=False)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig2 = go.Figure()
        colours_s = ['#f38ba8' if s > sigma_y else '#a6e3a1' for s in peak_stresses]
        fig2.add_trace(go.Scatter(
            x=voltages, y=peak_stresses/1e6,
            mode='lines', name='Peak Stress',
            line=dict(color='#fab387', width=2.5),
            fill='tozeroy', fillcolor='rgba(250,179,135,0.1)'
        ))
        fig2.add_hline(y=sigma_y/1e6, line_dash='dot', line_color='#f38ba8',
                       annotation_text=f"Yield at {sigma_y_:.0f} MPa")
        # Mark yield onset
        yield_v = voltages[peak_stresses >= sigma_y]
        if len(yield_v) > 0:
            fig2.add_vline(x=yield_v[0], line_color='#f38ba8', line_dash='dash',
                           annotation_text=f"Yield onset: {yield_v[0]:.0f}V")
        fig2.update_layout(
            title="Peak Stress vs Voltage — Yield Onset",
            xaxis_title="Voltage (V)", yaxis_title="Peak Stress (MPa)",
            template='plotly_dark', height=380
        )
        st.plotly_chart(fig2, use_container_width=True)

    # Energy efficiency
    col3, col4 = st.columns(2)
    with col3:
        efficiency = disps * 1e6 / (np.array(energies) * 1e9 + 1e-30)
        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(
            x=voltages, y=np.array(energies)*1e9,
            mode='lines', name='Stored Energy',
            line=dict(color='#cba6f7', width=2.5)
        ))
        fig3.update_layout(
            title="Electrostrictive Energy vs Voltage",
            xaxis_title="Voltage (V)", yaxis_title="Energy (nJ)",
            template='plotly_dark', height=320
        )
        st.plotly_chart(fig3, use_container_width=True)

    with col4:
        fig4 = go.Figure()
        fig4.add_trace(go.Scatter(
            x=voltages[1:], y=efficiency[1:],
            mode='lines', name='μm/nJ',
            line=dict(color='#f9e2af', width=2.5)
        ))
        fig4.update_layout(
            title="Actuation Efficiency (Displacement per Energy)",
            xaxis_title="Voltage (V)", yaxis_title="μm / nJ",
            template='plotly_dark', height=320
        )
        st.plotly_chart(fig4, use_container_width=True)


# ══════════════════════════════════════════════
# TAB 3 — MODAL ANALYSIS
# ══════════════════════════════════════════════
with tabs[2]:
    st.header("Modal Analysis — Natural Frequencies & Mode Shapes")

    n_modes_show = st.slider("Number of modes", 2, 8, 6, key='nmodes')

    with st.spinner("Computing modal solution..."):
        freqs, modes_full = fem_modal(E_, I_, L_, t_, b_, rho_, n_elem, n_modes_show)

    x_modal = np.linspace(0, L_, n_elem + 1)

    # Freq bar
    fig_freq = go.Figure()
    fig_freq.add_trace(go.Bar(
        x=[f"Mode {i+1}" for i in range(n_modes_show)],
        y=freqs,
        marker_color=px.colors.sequential.Plasma[:n_modes_show],
        text=[f"{f:.1f} Hz" for f in freqs],
        textposition='outside'
    ))
    fig_freq.update_layout(
        title="Natural Frequencies",
        yaxis_title="Frequency (Hz)",
        template='plotly_dark', height=300
    )
    st.plotly_chart(fig_freq, use_container_width=True)

    # Mode shape grid
    cols = st.columns(min(n_modes_show, 3))
    for i in range(n_modes_show):
        col = cols[i % 3]
        phi = modes_full[:, i]
        phi /= (np.max(np.abs(phi)) + 1e-30)
        with col:
            fig_m = go.Figure()
            fig_m.add_trace(go.Scatter(
                x=x_modal*1e3, y=phi,
                mode='lines',
                line=dict(color=px.colors.qualitative.Pastel[i % 10], width=2.5),
                fill='tozeroy', fillcolor=f'rgba(100,180,255,0.1)'
            ))
            fig_m.add_hline(y=0, line_color='white', line_width=0.5)
            fig_m.update_layout(
                title=f"Mode {i+1}: {freqs[i]:.1f} Hz",
                xaxis_title="mm", yaxis_title="Norm. Amplitude",
                template='plotly_dark', height=250,
                margin=dict(l=30, r=10, t=40, b=30)
            )
            st.plotly_chart(fig_m, use_container_width=True)

    # Analytical vs FEM comparison (Euler-Bernoulli)
    st.subheader("FEM vs Analytical (Euler-Bernoulli Cantilever)")
    beta_L = np.array([1.87510, 4.69409, 7.85476, 10.9955, 14.1372, 17.2788])[:n_modes_show]
    freqs_analytical = (beta_L**2 / (2*np.pi)) * np.sqrt(E_ * I_ / (rho_ * A_ * L_**4))
    comp_data = {
        "Mode": [f"Mode {i+1}" for i in range(n_modes_show)],
        "FEM (Hz)": [f"{f:.3f}" for f in freqs],
        "Analytical (Hz)": [f"{f:.3f}" for f in freqs_analytical],
        "Error (%)": [f"{abs(f-a)/a*100:.3f}" for f, a in zip(freqs, freqs_analytical)]
    }
    st.table(comp_data)


# ══════════════════════════════════════════════
# TAB 4 — PLASTICITY
# ══════════════════════════════════════════════
with tabs[3]:
    st.header("Plasticity Analysis — Von Mises Criterion")

    with st.spinner("Running plasticity analysis..."):
        _, _, x_mid, stress, _, _, *_ = fem_cantilever(
            V_, E_, I_, L_, t_, b_, M_, n_elem, rho_)

    s1 = stress
    s2 = np.zeros_like(s1)
    s3 = np.zeros_like(s1)
    sigma_vm = np.abs(s1)
    sv = (s1 + s2 + s3) / 3.0
    s_dev = s1 - sv
    phi = (s1 - s2)**2 + (s2 - s3)**2 + (s3 - s1)**2
    phi_yield = 2 * sigma_y**2

    col1, col2 = st.columns(2)

    # Von Mises field
    with col1:
        clrs = ['#f38ba8' if v >= sigma_y else '#89b4fa' for v in sigma_vm]
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=x_mid*1e3, y=sigma_vm/1e6,
            marker_color=clrs, name='σ_vm'
        ))
        fig.add_hline(y=sigma_y/1e6, line_dash='dot', line_color='#f38ba8',
                      annotation_text=f"σ_y = {sigma_y_} MPa")
        fig.update_layout(
            title="Von Mises Stress Field",
            xaxis_title="Position (mm)", yaxis_title="σ_vm (MPa)",
            template='plotly_dark', height=350
        )
        st.plotly_chart(fig, use_container_width=True)

    # π-plane
    with col2:
        xi_locus, eta_locus = yield_locus_pi_plane(sigma_y)
        xi_pts  = s1 / np.sqrt(2)
        eta_pts = s1 / np.sqrt(6)
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=xi_locus/1e6, y=eta_locus/1e6,
            mode='lines', name='Yield locus',
            line=dict(color='#f38ba8', width=2.5)
        ))
        fig2.add_trace(go.Scatter(
            x=xi_pts/1e6, y=eta_pts/1e6,
            mode='markers', name='Stress states',
            marker=dict(
                color=sigma_vm/1e6,
                colorscale='RdYlGn_r',
                size=8,
                colorbar=dict(title='σ_vm (MPa)')
            )
        ))
        fig2.update_layout(
            title="π-Plane Yield Locus",
            xaxis_title="ξ (MPa)", yaxis_title="η (MPa)",
            template='plotly_dark', height=350,
            yaxis_scaleanchor="x"
        )
        st.plotly_chart(fig2, use_container_width=True)

    # 3-D yield surface
    col3, col4 = st.columns(2)
    with col3:
        from physics.plasticity import yield_surface_cylinder
        X, Y, Z = yield_surface_cylinder(sigma_y)
        fig3 = go.Figure()
        fig3.add_trace(go.Surface(
            x=X/1e6, y=Y/1e6, z=Z/1e6,
            opacity=0.35,
            colorscale='Purples',
            showscale=False
        ))
        # Add stress path
        h_max = 1.5*sigma_y/np.sqrt(3)/1e6
        t_ax = np.linspace(-h_max, h_max, 50)
        fig3.add_trace(go.Scatter3d(
            x=t_ax, y=t_ax, z=t_ax,
            mode='lines', name='Hydrostatic axis',
            line=dict(color='white', width=2, dash='dash')
        ))
        # Add current stress points (s1, 0, 0)
        fig3.add_trace(go.Scatter3d(
            x=s1[::4]/1e6, y=np.zeros(len(s1[::4])),
            z=np.zeros(len(s1[::4])),
            mode='markers', name='Stress states',
            marker=dict(size=4, color='#f38ba8')
        ))
        fig3.update_layout(
            title="Von Mises Yield Cylinder",
            scene=dict(
                xaxis_title="σ₁ (MPa)",
                yaxis_title="σ₂ (MPa)",
                zaxis_title="σ₃ (MPa)",
                bgcolor='rgb(17,17,27)'
            ),
            template='plotly_dark', height=420
        )
        st.plotly_chart(fig3, use_container_width=True)

    # Volumetric vs Deviatoric
    with col4:
        fig4 = go.Figure()
        fig4.add_trace(go.Scatter(
            x=x_mid*1e3, y=sv/1e6,
            mode='lines', name='Hydrostatic σ_vol',
            line=dict(color='#89b4fa', width=2.5)
        ))
        fig4.add_trace(go.Scatter(
            x=x_mid*1e3, y=np.abs(s_dev)/1e6,
            mode='lines', name='|Deviatoric s_dev|',
            line=dict(color='#f38ba8', width=2.5)
        ))
        fig4.add_trace(go.Scatter(
            x=x_mid*1e3, y=sigma_vm/1e6,
            mode='lines', name='σ_vm',
            line=dict(color='#a6e3a1', width=1.5, dash='dash')
        ))
        fig4.add_hline(y=sigma_y/1e6, line_dash='dot', line_color='white')
        fig4.update_layout(
            title="Volumetric vs Deviatoric — Only Deviatoric Causes Yielding",
            xaxis_title="Position (mm)", yaxis_title="Stress (MPa)",
            template='plotly_dark', height=420
        )
        st.plotly_chart(fig4, use_container_width=True)

    # Distortion energy
    phi_norm = phi / (phi_yield + 1e-30)
    fig5 = go.Figure()
    fig5.add_trace(go.Scatter(
        x=x_mid*1e3, y=phi_norm,
        mode='lines+markers', name='φ/φ_yield',
        line=dict(color='#fab387', width=2.5),
        fill='tozeroy', fillcolor='rgba(250,179,135,0.1)'
    ))
    fig5.add_hline(y=1.0, line_dash='dot', line_color='#f38ba8',
                   annotation_text="Yield threshold")
    fig5.add_hrect(y0=1.0, y1=max(phi_norm.max()*1.1, 1.1),
                   fillcolor='rgba(243,139,168,0.15)',
                   annotation_text="Plastic region")
    fig5.update_layout(
        title="Distortion Energy Ratio φ / φ_yield",
        xaxis_title="Position (mm)", yaxis_title="φ / φ_yield",
        template='plotly_dark', height=300
    )
    st.plotly_chart(fig5, use_container_width=True)


# ══════════════════════════════════════════════
# TAB 5 — RESONANCE EXPLORER  ★ NOVEL
# ══════════════════════════════════════════════
with tabs[4]:
    st.header("🔬 Resonance Explorer — AC Voltage Excitation")
    st.markdown("""
    Apply a **sinusoidal voltage** V(t) = V₀ sin(2πft) and observe the beam's
    frequency response. Near a natural frequency the beam resonates — deflection
    amplifies dramatically. This is **not** in typical FEM textbooks.
    """)

    col_r1, col_r2 = st.columns([1, 2])
    with col_r1:
        V0_ac    = st.slider("AC Voltage Amplitude V₀ (V)", 10.0, 200.0, 50.0, 5.0)
        f_min    = st.slider("Sweep from (Hz)", 1.0, 500.0, 1.0, 1.0)
        f_max_r  = st.slider("Sweep to (Hz)", 50.0, 5000.0, 1000.0, 50.0)
        zeta     = st.slider("Damping ratio ζ", 0.001, 0.1, 0.02, 0.001)
        n_freq   = st.slider("Frequency points", 50, 500, 200, key='nfr')

    with st.spinner("Computing frequency response..."):
        freqs_nat, _ = fem_modal(E_, I_, L_, t_, b_, rho_, n_elem, n_modes=6)
        # SDOF FRF for each mode, summed
        f_sweep = np.linspace(f_min, f_max_r, n_freq)
        # Compute modal participation: mode 1 dominates tip deflection
        # Simplified FRF: H(f) = Σ_r  φ_r² / (ω_r² - ω² + 2iζω_r·ω)
        H_total = np.zeros(n_freq, dtype=complex)
        for fr in freqs_nat:
            wr = 2 * np.pi * fr
            w  = 2 * np.pi * f_sweep
            H_r = 1.0 / (wr**2 - w**2 + 2j * zeta * wr * w)
            H_total += H_r

        # Static deflection normalisation
        _, w_static, *_ = fem_cantilever(V0_ac, E_, I_, L_, t_, b_, M_, n_elem, rho_)
        w_tip_static = max(abs(w_static.max()), 1e-15)
        frf_mag = np.abs(H_total) * w_tip_static * freqs_nat[0]**2

    with col_r2:
        fig_frf = go.Figure()
        fig_frf.add_trace(go.Scatter(
            x=f_sweep, y=frf_mag*1e6,
            mode='lines', name='|H(f)|',
            line=dict(color='#89b4fa', width=2)
        ))
        for i, fn in enumerate(freqs_nat):
            fig_frf.add_vline(x=fn, line_dash='dash',
                              line_color='#f38ba8', line_width=1,
                              annotation_text=f"f{i+1}={fn:.0f}Hz",
                              annotation_font_size=10)
        fig_frf.update_xaxes(type='log' if f_max_r > 500 else 'linear')
        fig_frf.update_yaxes(type='log')
        fig_frf.update_layout(
            title=f"Frequency Response Function — Resonance Peaks at Natural Frequencies (ζ = {zeta})",
            xaxis_title="Excitation Frequency (Hz)",
            yaxis_title="Tip Amplitude (μm)  [log scale]",
            template='plotly_dark', height=420
        )
        st.plotly_chart(fig_frf, use_container_width=True)

    st.info(f"**Resonant frequencies:** "
            + "  |  ".join([f"f{i+1} = {f:.1f} Hz" for i, f in enumerate(freqs_nat)]))

    # Dynamic amplification factor
    st.subheader("Dynamic Amplification Factor")
    f_pick = st.select_slider("Pick excitation frequency",
                               options=[f"{v:.0f}" for v in f_sweep],
                               value=f"{f_sweep[len(f_sweep)//4]:.0f}")
    f_pick_v = float(f_pick)
    daf = np.interp(f_pick_v, f_sweep, frf_mag) / (w_tip_static + 1e-30)
    st.metric("Dynamic Amplification Factor (DAF)", f"{daf:.2f}×",
              delta="DANGER: resonance!" if daf > 5 else "Safe")


# ══════════════════════════════════════════════
# TAB 6 — MULTI-PATCH ACTUATOR  ★ NOVEL
# ══════════════════════════════════════════════
with tabs[5]:
    st.header("🎯 Multi-Patch Actuator Array — Shape Control")
    st.markdown("""
    Place up to **5 independent actuator patches** along the beam with individual voltages.
    See how their combined strain field shapes the beam — this is how real docking
    alignment correction systems work.
    """)

    n_patches = st.slider("Number of patches", 1, 5, 3)
    patches = []
    cols_p = st.columns(n_patches)
    for i, cp in enumerate(cols_p):
        with cp:
            x_start = st.slider(f"P{i+1} start (mm)", 0.0, float(L_mm-5), float(i*L_mm/n_patches), 1.0, key=f'ps{i}')
            x_end   = st.slider(f"P{i+1} end (mm)",   float(x_start+2), float(L_mm), float((i+1)*L_mm/n_patches), 1.0, key=f'pe{i}')
            Vp      = st.slider(f"P{i+1} voltage (V)", 0.0, 500.0, V_, 5.0, key=f'pv{i}')
            patches.append((x_start*1e-3, x_end*1e-3, Vp))

    with st.spinner("Solving multi-patch FEM..."):
        Le = L_ / n_elem
        ndof = 2 * (n_elem + 1)
        K = np.zeros((ndof, ndof)); F = np.zeros(ndof)
        eps_arr_mp = np.zeros(n_elem)

        for e in range(n_elem):
            from fem.beam_element import stiffness as bstiff
            ke = bstiff(E_, I_, Le)
            idx = [2*e, 2*e+1, 2*e+2, 2*e+3]
            K[np.ix_(idx, idx)] += ke
            x_e = (e + 0.5) * Le
            eps_total_e = 0.0
            for (xs, xe, Vp) in patches:
                if xs <= x_e <= xe:
                    eps_total_e += M_ * (Vp / t_) ** 2
            eps_arr_mp[e] = eps_total_e
            if eps_total_e > 0:
                from fem.beam_element import electrostrictive_equiv_force
                fe = electrostrictive_equiv_force(eps_total_e, E_, I_, Le)
                F[np.array(idx)] += fe

        K_ff, F_f, free_dofs, _ = apply_bc(K, F, [0, 1])
        u_mp = solve_static(K_ff, F_f, free_dofs, ndof)

    x_mp = np.linspace(0, L_, n_elem + 1)
    w_mp = u_mp[0::2]

    col_mp1, col_mp2 = st.columns(2)
    with col_mp1:
        fig_mp = go.Figure()
        # Patch regions
        for i, (xs, xe, Vp) in enumerate(patches):
            if Vp > 0:
                fig_mp.add_vrect(x0=xs*1e3, x1=xe*1e3,
                                  fillcolor=f'rgba(166,227,161,0.15)',
                                  line_width=0,
                                  annotation_text=f"P{i+1}:{Vp:.0f}V",
                                  annotation_position='top left')
        fig_mp.add_trace(go.Scatter(
            x=x_mp*1e3, y=w_mp*1e6,
            mode='lines', line=dict(color='#cba6f7', width=3),
            fill='tozeroy', fillcolor='rgba(203,166,247,0.12)'
        ))
        fig_mp.update_layout(
            title="Multi-Patch Actuated Shape",
            xaxis_title="Position (mm)", yaxis_title="Deflection (μm)",
            template='plotly_dark', height=380
        )
        st.plotly_chart(fig_mp, use_container_width=True)

    with col_mp2:
        fig_ep = go.Figure()
        x_elem_mid = np.array([(e+0.5)*Le for e in range(n_elem)])
        fig_ep.add_trace(go.Bar(
            x=x_elem_mid*1e3, y=eps_arr_mp,
            marker_color='#f9e2af', name='ε_electro per element'
        ))
        for i, (xs, xe, Vp) in enumerate(patches):
            if Vp > 0:
                fig_ep.add_vrect(x0=xs*1e3, x1=xe*1e3,
                                  fillcolor='rgba(166,227,161,0.1)',
                                  line_color='#a6e3a1', line_width=1)
        fig_ep.update_layout(
            title="Electrostrictive Strain Distribution from Patch Array",
            xaxis_title="Position (mm)", yaxis_title="ε_electro",
            template='plotly_dark', height=380
        )
        st.plotly_chart(fig_ep, use_container_width=True)

    c1, c2, c3 = st.columns(3)
    c1.metric("Tip Deflection", f"{w_mp.max()*1e6:.4f} μm")
    c2.metric("Peak Deflection Location", f"{x_mp[np.argmax(w_mp)]*1e3:.1f} mm")
    c3.metric("Active Patches", f"{sum(1 for _,_,v in patches if v > 0)}/{n_patches}")


# ══════════════════════════════════════════════
# TAB 7 — INVERSE PROBLEM  ★ NOVEL
# ══════════════════════════════════════════════
with tabs[6]:
    st.header("🧠 Inverse Problem — Find the Voltage for a Target Shape")
    st.markdown("""
    Instead of asking *"given V, what is δ?"* — ask the **inverse question**:
    *"I need a tip deflection of X μm — what voltage do I need?"*
    Or even harder: *"I need the beam to match a target shape — what patch voltages do I need?"*
    """)

    inv_tab1, inv_tab2 = st.tabs(["Single actuator", "Target shape (2 patches)"])

    with inv_tab1:
        target_disp = st.slider("Target tip deflection (μm)", 0.01, 2.0, 0.5, 0.01)

        # Analytical inversion: δ = C · ε = C · M (V/t)²  → V = t √(δ/(CM))
        # C from FEM: run at V=1 to get unit deflection coefficient
        _, w_unit, *_ = fem_cantilever(1.0, E_, I_, L_, t_, b_, M_, n_elem, rho_)
        C_unit = w_unit.max() / (M_ * (1.0 / t_) ** 2 + 1e-30)  # μ/ε_unit
        target_m = target_disp * 1e-6
        V_required = t_ * np.sqrt(target_m / (C_unit * M_ + 1e-30))

        st.metric("Required Voltage", f"{V_required:.2f} V",
                  delta="Exceeds V_max!" if V_required > V_max_ else "✅ Feasible")

        # Verify
        _, w_verify, _, s_verify, *_ = fem_cantilever(
            V_required, E_, I_, L_, t_, b_, M_, n_elem, rho_)
        x_inv = np.linspace(0, L_, n_elem + 1)

        fig_inv = go.Figure()
        fig_inv.add_trace(go.Scatter(
            x=x_inv*1e3, y=w_verify*1e6,
            mode='lines', name='FEM deflection',
            line=dict(color='#a6e3a1', width=3)
        ))
        fig_inv.add_hline(y=target_disp, line_dash='dot', line_color='#f9e2af',
                          annotation_text=f"Target: {target_disp} μm")
        fig_inv.update_layout(
            title=f"Verification: V = {V_required:.1f}V gives δ = {w_verify.max()*1e6:.4f} μm",
            xaxis_title="Position (mm)", yaxis_title="Deflection (μm)",
            template='plotly_dark', height=350
        )
        st.plotly_chart(fig_inv, use_container_width=True)

        # Check yield
        if np.any(np.abs(s_verify) >= sigma_y):
            st.error(f"⚠️ Yielding occurs at V = {V_required:.1f}V — target is not plastically safe!")
        else:
            st.success(f"✅ No yielding at V = {V_required:.1f}V — design is safe.")

    with inv_tab2:
        st.markdown("**Set a target tip-and-midpoint deflection and find the two patch voltages.**")
        col_a, col_b = st.columns(2)
        with col_a:
            d_mid_tgt = st.slider("Target mid-beam deflection (μm)", 0.0, 1.0, 0.3, 0.01)
        with col_b:
            d_tip_tgt = st.slider("Target tip deflection (μm)", 0.0, 2.0, 0.8, 0.01)

        from scipy.optimize import minimize

        def residual(Vs):
            V1, V2 = Vs
            if V1 < 0 or V2 < 0: return 1e6
            Le = L_ / n_elem
            ndof = 2 * (n_elem + 1)
            K = np.zeros((ndof, ndof)); F = np.zeros(ndof)
            for e in range(n_elem):
                from fem.beam_element import stiffness as bstiff, electrostrictive_equiv_force
                ke = bstiff(E_, I_, Le)
                idx = [2*e, 2*e+1, 2*e+2, 2*e+3]
                K[np.ix_(idx, idx)] += ke
                x_e = (e + 0.5) * Le
                Vp = V1 if x_e <= L_/2 else V2
                eps_e_v = M_ * (Vp / t_)**2
                fe = electrostrictive_equiv_force(eps_e_v, E_, I_, Le)
                F[np.array(idx)] += fe
            K_ff, F_f, fd, _ = apply_bc(K, F, [0, 1])
            u_s = solve_static(K_ff, F_f, fd, ndof)
            w_s = u_s[0::2]
            mid_idx = n_elem // 2
            e1 = (w_s[mid_idx]*1e6 - d_mid_tgt)**2
            e2 = (w_s[-1]*1e6 - d_tip_tgt)**2
            return e1 + e2

        with st.spinner("Solving inverse problem (optimisation)..."):
            from scipy.optimize import minimize
            res = minimize(residual, x0=[V_*0.7, V_], method='Nelder-Mead',
                           options={'xatol': 0.1, 'fatol': 1e-8, 'maxiter': 500})
            V1_opt, V2_opt = res.x

        st.metric("Patch 1 voltage (first half)", f"{V1_opt:.1f} V")
        st.metric("Patch 2 voltage (second half)", f"{V2_opt:.1f} V")
        st.caption(f"Optimiser residual: {res.fun:.2e}  |  Success: {res.success}")


# ══════════════════════════════════════════════
# TAB 8 — DESIGN OPTIMISER
# ══════════════════════════════════════════════
with tabs[7]:
    st.header("⚖️ Design Optimiser — Pareto Front")
    st.markdown("Maximise tip deflection while keeping stress below yield. Shows the **optimal (V, t) design space**.")

    n_V_opt = st.slider("V sweep points", 5, 30, 15, key='nvo')
    n_t_opt = st.slider("t sweep points", 5, 20, 12, key='nto')
    V_opt_range = st.slider("Voltage range (V)", 10.0, 800.0, (20.0, float(V_max_)), key='vor')
    t_opt_range = st.slider("Thickness range (mm)", 0.3, 5.0, (0.3, 2.5), key='tor')

    with st.spinner("Running design sweep..."):
        Vs_opt = np.linspace(V_opt_range[0], V_opt_range[1], n_V_opt)
        ts_opt = np.linspace(t_opt_range[0]*1e-3, t_opt_range[1]*1e-3, n_t_opt)
        VV, TT = np.meshgrid(Vs_opt, ts_opt, indexing='ij')
        DISP_g = np.zeros_like(VV)
        STRESS_g = np.zeros_like(VV)

        for i, Vo in enumerate(Vs_opt):
            for j, to in enumerate(ts_opt):
                Io = b_ * to**3 / 12
                _, wo, _, so, *_ = fem_cantilever(
                    Vo, E_, Io, L_, to, b_, M_, n_elem, rho_)
                DISP_g[i,j]   = wo.max()
                STRESS_g[i,j] = np.abs(so).max()

    col_o1, col_o2 = st.columns(2)
    with col_o1:
        fig_d = go.Figure(data=go.Contour(
            x=Vs_opt, y=ts_opt*1e3, z=DISP_g.T*1e6,
            colorscale='Viridis',
            colorbar=dict(title='Max δ (μm)')
        ))
        fig_d.update_layout(
            title="Tip Deflection Landscape",
            xaxis_title="Voltage (V)", yaxis_title="Thickness (mm)",
            template='plotly_dark', height=380
        )
        st.plotly_chart(fig_d, use_container_width=True)

    with col_o2:
        feasible_g = STRESS_g < sigma_y
        fig_s = go.Figure(data=go.Contour(
            x=Vs_opt, y=ts_opt*1e3, z=STRESS_g.T/1e6,
            colorscale='RdYlGn_r',
            colorbar=dict(title='Max σ (MPa)')
        ))
        # Yield boundary contour
        fig_s.add_trace(go.Contour(
            x=Vs_opt, y=ts_opt*1e3,
            z=feasible_g.T.astype(float),
            contours_coloring='lines',
            line_width=3,
            colorscale=[[0,'rgba(0,0,0,0)'],[1,'rgba(0,0,0,0)']],
            contours=dict(start=0.5, end=0.5, size=1),
            showscale=False, name='Yield boundary',
            line_color='white'
        ))
        fig_s.update_layout(
            title="Stress Landscape  (white = yield boundary)",
            xaxis_title="Voltage (V)", yaxis_title="Thickness (mm)",
            template='plotly_dark', height=380
        )
        st.plotly_chart(fig_s, use_container_width=True)

    # Pareto front
    pareto_t, pareto_d, pareto_V = [], [], []
    for j, to in enumerate(ts_opt):
        best_d, best_V = 0, 0
        for i, Vo in enumerate(Vs_opt):
            if STRESS_g[i,j] < sigma_y and DISP_g[i,j] > best_d:
                best_d = DISP_g[i,j]; best_V = Vo
        if best_d > 0:
            pareto_t.append(to*1e3)
            pareto_d.append(best_d*1e6)
            pareto_V.append(best_V)

    if pareto_d:
        fig_p = go.Figure()
        fig_p.add_trace(go.Scatter(
            x=pareto_t, y=pareto_d,
            mode='lines+markers',
            marker=dict(color=pareto_V, colorscale='Plasma', size=12,
                        showscale=True, colorbar=dict(title='Optimal V (V)')),
            line=dict(color='lightgray', width=1.5),
            text=[f"V={v:.0f}V" for v in pareto_V],
            hovertemplate="%{text}<extra></extra>"
        ))
        fig_p.update_layout(
            title="Pareto Front — Maximum Safe Deflection for Each Thickness",
            xaxis_title="Thickness (mm)",
            yaxis_title="Max Deflection within Yield Limit (μm)",
            template='plotly_dark', height=350
        )
        st.plotly_chart(fig_p, use_container_width=True)


# ══════════════════════════════════════════════
# TAB 9 — 2D PLATE
# ══════════════════════════════════════════════
with tabs[8]:
    st.header("2-D Plate FEM (CST Elements)")

    col_pl1, col_pl2 = st.columns([1, 2])
    with col_pl1:
        Lx_mm = st.slider("Plate Lx (mm)", 20.0, 200.0, 100.0, 5.0)
        Ly_mm = st.slider("Plate Ly (mm)", 10.0, 100.0, 50.0, 5.0)
        nx_p  = st.slider("Divisions x", 4, 20, 8)
        ny_p  = st.slider("Divisions y", 2, 10, 4)
        st.caption(f"Elements: {nx_p*ny_p*2}  |  Nodes: {(nx_p+1)*(ny_p+1)}")

    Lx_ = Lx_mm * 1e-3; Ly_ = Ly_mm * 1e-3

    with st.spinner("Assembling 2-D plate FEM..."):
        D = D_matrix_2D_plane_stress(E_, nu_)
        nodes_p, elements_p = mesh_rect_cst(Lx_, Ly_, nx_p, ny_p)
        n_nodes_p = nodes_p.shape[0]
        ndof_p    = 2 * n_nodes_p

        K_p, Bs_p, As_p = assemble_plate(nodes_p, elements_p, D, thickness=t_)

        eps_e_p = M_ * (V_ / t_) ** 2
        F_p = np.zeros(ndof_p)
        eps_eigen = np.array([eps_e_p, 0.0, 0.0])
        for ie, elem in enumerate(elements_p):
            B = Bs_p[ie]
            if B is None: continue
            f_eq = B.T @ D @ eps_eigen * As_p[ie] * t_
            dofs = []
            for nd in elem: dofs += [2*nd, 2*nd+1]
            F_p[dofs] += f_eq

        left_nodes = np.where(nodes_p[:, 0] < 1e-9)[0]
        fixed_p = []
        for nd in left_nodes: fixed_p += [2*nd, 2*nd+1]
        free_p = np.setdiff1d(np.arange(ndof_p), fixed_p)
        K_ff_p = K_p[np.ix_(free_p, free_p)]
        F_f_p  = F_p[free_p]
        u_free_p = np.linalg.solve(K_ff_p, F_f_p)
        u_p = np.zeros(ndof_p)
        u_p[free_p] = u_free_p

        ux_p = u_p[0::2]; uy_p = u_p[1::2]
        mag_p = np.sqrt(ux_p**2 + uy_p**2)

        node_vm = np.zeros(n_nodes_p); node_cnt = np.zeros(n_nodes_p)
        for ie, elem in enumerate(elements_p):
            B = Bs_p[ie]
            if B is None: continue
            dofs = []
            for nd in elem: dofs += [2*nd, 2*nd+1]
            u_e = u_p[dofs]
            sigma, _ = cst_stress(u_e, B, D, eps_eigen=eps_eigen)
            from physics.plasticity import von_mises_from_components
            vm = von_mises_from_components(sigma[0], sigma[1], sxy=sigma[2])
            for nd in elem:
                node_vm[nd] += vm; node_cnt[nd] += 1
        node_cnt = np.maximum(node_cnt, 1)
        node_vm /= node_cnt

    with col_pl2:
        fig_pl = make_subplots(rows=1, cols=2,
                                subplot_titles=["x-Displacement (μm)", "Von Mises Stress (MPa)"])
        fig_pl.add_trace(go.Scatter(
            x=nodes_p[:,0]*1e3, y=nodes_p[:,1]*1e3,
            mode='markers',
            marker=dict(color=ux_p*1e6, colorscale='RdBu', size=8,
                        colorbar=dict(title='ux (μm)', x=0.45)),
        ), row=1, col=1)
        fig_pl.add_trace(go.Scatter(
            x=nodes_p[:,0]*1e3, y=nodes_p[:,1]*1e3,
            mode='markers',
            marker=dict(color=node_vm/1e6, colorscale='Jet', size=8,
                        colorbar=dict(title='σ_vm (MPa)', x=1.01)),
        ), row=1, col=2)
        fig_pl.update_layout(template='plotly_dark', height=380,
                              showlegend=False)
        st.plotly_chart(fig_pl, use_container_width=True)

    # Deformed mesh
    scale_def = st.slider("Deformation scale factor", 100, 100000, 10000, 100)
    nodes_def = nodes_p + scale_def * np.column_stack([ux_p, uy_p])

    fig_dm = go.Figure()
    for elem in elements_p[::max(1, len(elements_p)//200)]:   # subsample for speed
        pt_o = nodes_p[np.append(elem, elem[0])]
        pt_d = nodes_def[np.append(elem, elem[0])]
        fig_dm.add_trace(go.Scatter(x=pt_o[:,0]*1e3, y=pt_o[:,1]*1e3,
                                     mode='lines', line=dict(color='#89b4fa', width=0.5),
                                     showlegend=False))
        fig_dm.add_trace(go.Scatter(x=pt_d[:,0]*1e3, y=pt_d[:,1]*1e3,
                                     mode='lines', line=dict(color='#f38ba8', width=0.8),
                                     showlegend=False))
    fig_dm.update_layout(
        title=f"Deformed Mesh (scale ×{scale_def})",
        xaxis_title="x (mm)", yaxis_title="y (mm)",
        template='plotly_dark', height=320,
        xaxis_scaleanchor="y"
    )
    st.plotly_chart(fig_dm, use_container_width=True)

    c1, c2, c3 = st.columns(3)
    c1.metric("Max |u| displacement", f"{mag_p.max()*1e6:.4f} μm")
    c2.metric("Max σ_vm", f"{node_vm.max()/1e6:.2f} MPa")
    c3.metric("Yielded nodes",
              f"{int(np.sum(node_vm >= sigma_y))}/{n_nodes_p}")

# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────
st.divider()
st.caption("AE246 Structures Lab · IIT Bombay · Electrostrictive FEM Engine · Built with Streamlit + Plotly")
