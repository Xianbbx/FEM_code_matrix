# GUI Documentation — Electrostrictive FEM Lab

**File**: `gui.py`  
**Framework**: [Streamlit](https://streamlit.io) + [Plotly](https://plotly.com/python/)  
**Launch**: `streamlit run gui.py`

The GUI provides a fully interactive, dark-themed web application for exploring every simulation in the codebase. All FEM solves are cached with `@st.cache_data` so parameter changes only re-run what has changed.

---

## Global Sidebar Controls

The sidebar feeds every tab — any change immediately reruns the active tab's FEM solve.

### Material & Geometry

| Slider | Range | Default | Effect |
|---|---|---|---|
| Young's Modulus E | 10 – 300 GPa | 70 GPa | Scales all stiffness matrices $\mathbf{K} \propto E$ |
| Yield Stress $\sigma_y$ | 50 – 800 MPa | 250 MPa | Yield threshold for red/green colour coding |
| Poisson's Ratio $\nu$ | 0.10 – 0.49 | 0.33 | Used in plate 2-D $\mathbf{D}$ matrix only |
| Beam Length L | 20 – 300 mm | 100 mm | Changes element length $L_e = L/n$ |
| Thickness t | 0.2 – 5.0 mm | 1.0 mm | Changes $I = bt^3/12$ and $\varepsilon_e = M(V/t)^2$ |
| Width b | 1.0 – 20 mm | 5.0 mm | Changes $I$ and cross-sectional area $A$ |
| Electrostrictive coeff M | 0.1 – 50 ×10⁻¹⁸ m²/V² | 1.0 | Scales all electrostrictive strains |
| FEM Elements | 10 – 80 | 40 | Mesh density; more elements = higher accuracy but slower |

### Voltage Control

| Slider | Default | Purpose |
|---|---|---|
| Applied Voltage V | 100 V | Single operating point used in static analyses |
| Sweep Max Voltage | 300 V | Upper limit for all voltage sweep plots |

> **Derived quantities** computed once from the sidebar and reused by all tabs:
> beam second moment $I = bt^3/12$, cross-section area $A = bt$, element length $L_e = L/n$, total DOFs $= 2(n+1)$.

---

## Tab 1 — Beam Analysis

**Purpose**: Single-voltage static FEM solve. The primary workbench tab.

### What Is Computed

1. `fem_cantilever(V, E, I, L, t, b, M, n_elem, rho)` — assembles and solves the beam FEM:
   - Global stiffness $\mathbf{K}$ and force $\mathbf{F}$ assembled from electrostrictive equivalent moments
   - Boundary conditions: DOFs 0 and 1 fixed (clamped root)
   - Static solve: $\mathbf{K}_{ff}\,\mathbf{u}_f = \mathbf{F}_f$
   - Stress recovery at each element midpoint

### Metric Row (4 columns)

| Metric | Formula | Units |
|---|---|---|
| Tip Deflection | $w_{\mathrm{tip}} = \max(u[0::2])$ | µm |
| Electrostrictive Strain | $\varepsilon_e = M(V/t)^2$ | dimensionless |
| Peak Bending Stress | $\max|\sigma|$ over all elements | MPa |
| Yielded Elements | count of elements where $|\sigma| \geq \sigma_y$ | n / total |

### Plots

**Deflection Profile** (top-left)  
Filled line chart: $w(x)$ in µm vs. position in mm. Blue fill shows the deformed shape. Includes a zero-baseline dashed line.

**Bending Stress Distribution** (top-right)  
Bar chart: $\sigma_{xx}$ in MPa per element. Bars are **red** where $|\sigma| \geq \sigma_y$ (yielded) and **green** (elastic). Dotted lines mark $\pm\sigma_y$.

**Curvature Profile** (bottom-left)  
Line chart of $\kappa(x)$ in 1/m per element. Computed from the Euler-Bernoulli B matrix at each element midpoint.

**Stress Through Thickness** (bottom-right)  
Linear stress distribution $\sigma(z) = \sigma_{\mathrm{top}} \cdot (z/c)$ through the depth of the highest-stress element. Shows whether the extreme fibre has yielded.

---

## Tab 2 — Voltage Sweep

**Purpose**: Validate $\varepsilon \propto V^2$, find yield onset voltage, compute energy efficiency.

### What Is Computed

A sweep over `n_sweep` voltage points from 0 to $V_{\mathrm{max}}$. For each voltage:
- Tip deflection $w_{\mathrm{tip}}$
- Peak bending stress $\max|\sigma|$
- Simplified stored electrostrictive energy $U = \frac{1}{2}E\varepsilon_e^2 A L$

### Controls

| Slider | Purpose |
|---|---|
| Number of sweep points | Trades speed vs. curve smoothness (10–100) |

### Plots

**Deflection vs V²** (top-left, dual-axis)  
Plots $w_{\mathrm{tip}}$ against $V^2$ rather than $V$ — if electrostrictive physics is correct the relationship is **linear** in this plot. A dashed theoretical linear fit is overlaid to confirm the $\varepsilon \propto V^2$ law.

**Peak Stress vs Voltage** (top-right)  
Filled area chart of $\sigma_{\mathrm{max}}$ vs. $V$. A dotted yield line at $\sigma_y$ and a vertical dashed line marking the **yield onset voltage** (first voltage where $\sigma_{\mathrm{max}} \geq \sigma_y$) are automatically added.

**Electrostrictive Energy vs Voltage** (bottom-left)  
Energy in nanojoules vs. voltage. Confirms $U \propto V^4$ (since $\varepsilon \propto V^2$ and $U \propto \varepsilon^2$).

**Actuation Efficiency** (bottom-right)  
Displacement-per-energy in µm/nJ vs. voltage. Shows the efficiency plateau — higher voltage gives more displacement but proportionally more energy, eventually becoming less efficient per unit energy.

---

## Tab 3 — Modal Analysis

**Purpose**: Compute natural frequencies and mode shapes; compare FEM against the analytical Euler-Bernoulli cantilever solution.

### What Is Computed

`fem_modal(E, I, L, t, b, rho, n_elem, n_modes)` — assembles both $\mathbf{K}$ and $\mathbf{M}$ (consistent mass), applies BCs, and solves:

$$\mathbf{K}_{ff}\,\phi = \lambda\,\mathbf{M}_{ff}\,\phi \quad \Rightarrow \quad f_n = \frac{\sqrt{\lambda_n}}{2\pi}$$

### Controls

| Slider | Purpose |
|---|---|
| Number of modes | 2–8; controls how many mode shapes to display |

### Plots

**Natural Frequency Bar Chart**  
Plasma-coloured bars for each mode with frequency labels above each bar.

**Mode Shape Grid** (3-column layout)  
One chart per mode, normalised so $\max|\phi_n| = 1$. Mode 1 is a smooth half-arch; each subsequent mode adds one more inflection point.

**FEM vs Analytical Table**  
Analytical Euler-Bernoulli frequencies use characteristic values $\beta_n L = 1.875, 4.694, 7.855, \ldots$:

$$f_n^{\mathrm{analytical}} = \frac{(\beta_n L)^2}{2\pi L^2}\sqrt{\frac{EI}{\rho A}}$$

The table shows Mode, FEM frequency, analytical frequency, and percentage error. Error drops as `n_elem` increases.

---

## Tab 4 — Plasticity

**Purpose**: Full Von Mises plasticity visualisation — stress field, pi-plane, 3-D yield cylinder, volumetric/deviatoric split, distortion energy.

### What Is Computed

Reuses `fem_cantilever` at the current sidebar voltage. The 1-D beam bending state gives $\sigma_1 = \sigma_{xx}$, $\sigma_2 = \sigma_3 = 0$, so:
- $\sigma_{\mathrm{vm}} = |\sigma_1|$
- Hydrostatic: $\sigma_{\mathrm{vol}} = \sigma_1/3$
- Deviatoric: $s = \sigma_1 - \sigma_{\mathrm{vol}} = 2\sigma_1/3$
- Distortion energy: $\varphi = (\sigma_1-\sigma_2)^2 + (\sigma_2-\sigma_3)^2 + (\sigma_3-\sigma_1)^2 = 2\sigma_1^2$
- Yield distortion energy threshold: $\varphi_{\mathrm{yield}} = 2\sigma_y^2$

### Plots

**Von Mises Stress Field** (top-left)  
Bar chart of $\sigma_{\mathrm{vm}}$ per element. Red bars = yielded. Dotted yield line at $\sigma_y$.

**Pi-Plane Yield Locus** (top-right)  
Scatter plot on the deviatoric $(\xi, \eta)$ plane. The red circle is the Von Mises yield locus $r = \sigma_y\sqrt{2/3}$. Each element's stress state is plotted as a coloured dot (colour = $\sigma_{\mathrm{vm}}$). Points outside the circle have yielded.

The pi-plane coordinates are:
$$\xi = \frac{\sigma_1}{\sqrt{2}}, \quad \eta = \frac{\sigma_1}{\sqrt{6}}$$
(since $\sigma_2 = \sigma_3 = 0$, all points fall on a straight line from the origin.)

**3-D Von Mises Yield Cylinder** (bottom-left)  
Interactive 3-D surface plot. The cylinder is semi-transparent purple; the hydrostatic axis (white dashed) runs through it diagonally; red dots show the current stress state at each element ($\sigma_1 \neq 0$, $\sigma_2 = \sigma_3 = 0$, so all points lie along the $\sigma_1$ axis).

**Volumetric vs. Deviatoric** (bottom-right)  
Line chart showing the hydrostatic stress $\sigma_{\mathrm{vol}}$, deviatoric $|s|$, and $\sigma_{\mathrm{vm}}$ along the beam. Since $\sigma_2 = \sigma_3 = 0$, the deviatoric stress equals $\frac{2}{3}\sigma_1$ and the hydrostatic is $\frac{1}{3}\sigma_1$.  
Key message: **only the deviatoric component (equal to $\sigma_{\mathrm{vm}}$) triggers yielding**.

**Distortion Energy Ratio** (full width)  
Bar chart of $\varphi / \varphi_{\mathrm{yield}}$. A dotted line at 1.0 marks the yield threshold. A pink shaded region highlights the plastic zone.

---

## Tab 5 — Resonance Explorer

**Purpose**: Simulate the frequency response of the beam to a sinusoidally varying voltage $V(t) = V_0 \sin(2\pi f t)$.

### Physics

Near a natural frequency $f_n$, the beam resonates — deflection amplifies dramatically. The Frequency Response Function (FRF) is computed using a modal superposition of single-DOF FRFs:

$$H(f) = \sum_{r=1}^{N} \frac{1}{\omega_r^2 - \omega^2 + 2i\zeta\omega_r\omega}$$

where $\omega_r = 2\pi f_r$ and $\zeta$ is the damping ratio.

### Controls

| Slider | Purpose |
|---|---|
| AC Voltage Amplitude $V_0$ | Sets the actuation level |
| Sweep from / to (Hz) | Frequency range of the FRF |
| Damping ratio $\zeta$ | Controls sharpness of resonance peaks |
| Frequency points | Resolution of the sweep |

### Plots

**Frequency Response Function**  
Log-scale amplitude of $|H(f)|$ vs. excitation frequency. Vertical dashed lines mark each natural frequency. Resonance peaks appear at $f_1, f_2, \ldots$

**Dynamic Amplification Factor (DAF)**  
A `select_slider` lets the user pick any excitation frequency. The DAF is:

$$\mathrm{DAF} = \frac{|H(f)|}{w_{\mathrm{tip, static}}}$$

A warning is shown if DAF > 5 (dangerous resonance condition).

---

## Tab 6 — Multi-Patch Actuator

**Purpose**: Place up to 5 independent actuator patches along the beam, each with its own voltage and coverage range. Observe combined deformed shape.

### Physics

Each element's electrostrictive strain is the sum of contributions from all active patches covering that element's midpoint:

$$\varepsilon_e^{(e)} = \sum_{\mathrm{patch}\, k} M\left(\frac{V_k}{t}\right)^2 \quad \text{if element midpoint} \in [x_{\mathrm{start},k},\, x_{\mathrm{end},k}]$$

The FEM then proceeds as normal: equivalent nodal moments from each element's $\varepsilon_e$ are assembled and solved.

### Controls (per patch)

| Slider | Purpose |
|---|---|
| P_n start (mm) | Left edge of patch coverage |
| P_n end (mm) | Right edge of patch coverage |
| P_n voltage (V) | Individual patch driving voltage |

### Plots

**Multi-Patch Actuated Shape**  
Deflection $w(x)$ with shaded regions showing each active patch's coverage and voltage label.

**Strain Distribution**  
Bar chart of $\varepsilon_e$ per element. Patch boundaries shown as coloured overlays. Elements with multiple overlapping patches accumulate strain.

### Metrics

- Tip Deflection
- Peak Deflection Location (mm from root)
- Active Patches count

---

## Tab 7 — Inverse Problem

**Purpose**: Instead of "given V, find deflection", ask the reverse: "given a target deflection, find the required voltage."

Two sub-tabs are provided:

---

### Sub-tab A: Single Actuator

**Question**: What voltage produces a target tip deflection of $\delta_{\mathrm{target}}$?

**Method**: Analytical inversion. A unit-voltage FEM solve gives the deflection coefficient $C$ (deflection per unit eigen-strain). Then:

$$V_{\mathrm{required}} = t \sqrt{\frac{\delta_{\mathrm{target}}}{C \cdot M}}$$

The result is fed back into the FEM for **verification**, and the deflection profile is plotted against the target line.

A yield check is also performed: if $V_{\mathrm{required}}$ causes yielding, a red warning is shown; otherwise a green success message.

| Slider | Purpose |
|---|---|
| Target tip deflection (µm) | Desired output deflection |

---

### Sub-tab B: Target Shape (2 patches)

**Question**: What are the voltages for a first-half and second-half patch that match both a midpoint deflection and a tip deflection simultaneously?

**Method**: Numerical optimisation using `scipy.optimize.minimize` (Nelder-Mead). The cost function is:

$$R(V_1, V_2) = \left(w_{\mathrm{mid}}(V_1, V_2) - \delta_{\mathrm{mid,target}}\right)^2 + \left(w_{\mathrm{tip}}(V_1, V_2) - \delta_{\mathrm{tip,target}}\right)^2$$

The optimiser finds $(V_1, V_2)$ that minimises $R$.

| Slider | Purpose |
|---|---|
| Target mid-beam deflection (µm) | Constraint at $x = L/2$ |
| Target tip deflection (µm) | Constraint at $x = L$ |

---

## Tab 8 — Design Optimiser

**Purpose**: Map the full 2-D design space $(V, t)$ and extract the **Pareto front** — the maximum safe deflection achievable at each thickness.

### What Is Computed

A grid sweep over `n_V × n_t` design points. For each $(V_i, t_j)$:
- Recomputes $I = bt_j^3/12$
- Runs the full FEM
- Records $w_{\mathrm{tip}}$ and $\sigma_{\mathrm{max}}$
- A point is **feasible** if $\sigma_{\mathrm{max}} < \sigma_y$

### Controls

| Slider | Purpose |
|---|---|
| V sweep points | Voltage grid resolution |
| t sweep points | Thickness grid resolution |
| Voltage range | Min and max voltage for the sweep |
| Thickness range | Min and max thickness for the sweep |

### Plots

**Tip Deflection Landscape** (contour)  
Filled contour map of $w_{\mathrm{tip}}$ on the $(V, t)$ plane. Hotter colors = larger deflection.

**Stress Landscape + Yield Boundary** (contour)  
Filled contour of $\sigma_{\mathrm{max}}$. A **white contour line** marks the yield boundary ($\sigma_{\mathrm{max}} = \sigma_y$). Points to the left/below are elastic; points above/right are plastic.

**Pareto Front**  
Scatter + line plot: for each thickness, the maximum feasible deflection (highest $V$ that stays elastic) is plotted. Marker colour shows the corresponding optimal voltage. This reveals the optimal operating thickness.

---

## Tab 9 — 2-D Plate FEM

**Purpose**: Run a full 2-D plane-stress CST FEM on a rectangular plate and visualise the 2-D displacement and Von Mises stress fields.

### What Is Computed

1. `D_matrix_2D_plane_stress(E, nu)` — builds the $3\times3$ plane-stress constitutive matrix
2. `mesh_rect_cst(Lx, Ly, nx, ny)` — generates nodes and triangular elements
3. `assemble_plate(nodes, elements, D, t)` — assembles global $\mathbf{K}$
4. Equivalent nodal forces from electrostrictive eigen-strain $\varepsilon_{xx} = M(V/t)^2$:
   $$f_{\mathrm{eq}}^{(e)} = \mathbf{B}^T \mathbf{D}\,\varepsilon^* \cdot A_e \cdot t$$
5. BCs: all nodes on the left edge ($x = 0$) are fully clamped ($u = v = 0$)
6. Stress recovery per element: $\sigma = \mathbf{D}(\mathbf{B}\mathbf{u}_e - \varepsilon^*)$, then Von Mises $\sigma_{\mathrm{vm}}$ averaged to nodes

### Controls (in-panel)

| Slider | Purpose |
|---|---|
| Plate Lx (mm) | Plate length |
| Plate Ly (mm) | Plate height |
| Divisions x / y | Mesh density (each quad = 2 triangles) |
| Deformation scale factor | Multiplier for deformation visualisation (100 – 100 000) |

### Plots

**x-Displacement and Von Mises Stress** (side by side scatter)  
Nodes coloured by $u_x$ (red-blue diverging scale) and by $\sigma_{\mathrm{vm}}$ (Jet scale). Clearly shows the plate extending in $x$ due to the electrostrictive strain in $\varepsilon_{xx}$.

**Deformed Mesh Overlay**  
Original mesh (blue) and deformed mesh (red) with adjustable scale factor. The deformation is exaggerated to make the shape visible at the micro-scale displacements involved.

### Metrics

- Max total displacement $|u| = \sqrt{u_x^2 + u_y^2}$ in µm
- Max nodal $\sigma_{\mathrm{vm}}$ in MPa
- Number of yielded nodes (where $\sigma_{\mathrm{vm}} \geq \sigma_y$)

---

## Caching Strategy

The two core FEM functions are decorated with `@st.cache_data`:

```python
@st.cache_data(show_spinner=False)
def fem_cantilever(V, E, I, L, t, b, M_coeff, n_elem, rho): ...

@st.cache_data(show_spinner=False)
def fem_modal(E, I, L, t, b, rho, n_elem, n_modes): ...
```

Streamlit caches by function arguments, so changing any sidebar slider only re-runs FEM solves whose inputs changed. Multiple tabs sharing the same parameter set reuse the cached result with zero recomputation cost.

---

## Summary — Tab vs. Physics Mapping

| Tab | Core Physics | FEM Type | Novel? |
|---|---|---|---|
| 1 Beam Analysis | Static deflection, bending stress, curvature | Beam (E-B) | Standard |
| 2 Voltage Sweep | $\varepsilon \propto V^2$ validation, yield onset, energy | Beam (E-B) | Standard |
| 3 Modal Analysis | $\mathbf{K}\phi = \lambda\mathbf{M}\phi$, natural frequencies | Beam (E-B) + mass | Standard |
| 4 Plasticity | Von Mises, pi-plane, yield cylinder, distortion energy | Beam (E-B) | Standard |
| 5 Resonance Explorer | FRF, dynamic amplification, damping | Modal superposition | **Novel** |
| 6 Multi-Patch Actuator | Spatially distributed independent actuators | Beam (E-B) | **Novel** |
| 7 Inverse Problem | Analytical inversion + Nelder-Mead optimisation | Beam (E-B) | **Novel** |
| 8 Design Optimiser | Pareto front in $(V, t)$ design space | Beam (E-B) | **Novel** |
| 9 2-D Plate | Plane-stress CST, 2-D displacement + stress field | CST plate | Standard |
