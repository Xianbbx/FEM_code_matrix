# Physics in the FEM Codebase — Complete Explanation

This document describes every piece of physics implemented across the codebase, from the underlying constitutive laws to the simulation scenarios.

---

## 1. Linear Elasticity (`physics/elasticity.py`)

### Constitutive Law (Hooke's Law)

The fundamental relationship between stress and strain is:

$$\sigma = \mathbf{D} \, (\varepsilon - \varepsilon^{*})$$

where $\mathbf{D}$ is the constitutive (material stiffness) matrix and $\varepsilon^{*}$ is any **eigen-strain** (a stress-free strain such as thermal expansion or electrostrictive strain).

### Plane Stress — `D_matrix_2D_plane_stress`

Used for thin plates where the out-of-plane stress $\sigma_{zz} = 0$. In Voigt notation $[\varepsilon_{xx},\, \varepsilon_{yy},\, \gamma_{xy}]$:

$$\mathbf{D}^{\mathrm{ps}} = \frac{E}{1-\nu^2}
\begin{bmatrix}
1   & \nu & 0 \\
\nu & 1   & 0 \\
0   & 0   & \dfrac{1-\nu}{2}
\end{bmatrix}$$

### Plane Strain — `D_matrix_2D_plane_strain`

Used for thick bodies where out-of-plane strain $\varepsilon_{zz} = 0$:

$$\mathbf{D}^{\mathrm{pe}} = \frac{E}{(1+\nu)(1-2\nu)}
\begin{bmatrix}
1-\nu & \nu   & 0 \\
\nu   & 1-\nu & 0 \\
0     & 0     & \dfrac{1-2\nu}{2}
\end{bmatrix}$$

### 1-D and Voigt Stress

The 1-D form $\sigma = E(\varepsilon - \varepsilon^{*})$ is used for beam elements; the general form $\sigma = \mathbf{D}(\varepsilon - \varepsilon^{*})$ handles 2-D/3-D cases.

### Principal Stresses

**2-D** — found analytically from $(\sigma_{xx},\, \sigma_{yy},\, \sigma_{xy})$:

$$\sigma_{1,2} = \frac{\sigma_{xx}+\sigma_{yy}}{2} \pm \sqrt{\left(\frac{\sigma_{xx}-\sigma_{yy}}{2}\right)^2 + \sigma_{xy}^2}$$

**3-D** — obtained as eigenvalues of the symmetric $3 \times 3$ stress tensor.

---

## 2. Electrostriction (`physics/electrostriction.py`)

Electrostriction is the deformation of a dielectric material under an applied electric field. Unlike piezoelectricity, the strain is proportional to the **square** of the electric field, making it inherently nonlinear.

### Uniform Electrostrictive Strain

For an actuator of thickness $t$ driven by voltage $V$, the electric field is $E_{\mathrm{field}} = V/t$, giving:

$$\varepsilon_e = M \left(\frac{V}{t}\right)^2$$

$M$ is the **electrostrictive coefficient** $[\mathrm{m^2 / V^2}]$, set to $10^{-18}\ \mathrm{m^2/V^2}$ for the default material. The $V^2$ dependence means tip deflection vs. voltage is a **parabola**, validated in the beam-bending simulation.

### Spatially Varying (Sinusoidal) Strain

For non-uniform actuation along a beam of length $L$:

$$\varepsilon_e(x) = M \left(\frac{V}{t}\right)^2 \sin\left(\frac{\pi x}{L}\right)$$

This models graded electrode coverage, producing a wavelike curvature distribution along the beam.

### Electrostrictive Energy

Simplified stored energy in the actuator, by analogy with $\frac{1}{2}\sigma\varepsilon$ elastic strain energy:

$$U = \frac{\varepsilon_e^2 \cdot V_{\mathrm{vol}}}{2M}$$

where $V_{\mathrm{vol}}$ is the material volume.

### Equivalent Bending Moment (Actuation Mechanism)

When an electrostrictive film is bonded to the **top surface** of a beam at $z = +h/2$, it imposes an eigen-curvature:

$$\kappa_e = \frac{\varepsilon_e}{h/2} = \frac{2\varepsilon_e}{t}$$

Via virtual work, this generates equivalent nodal forces — a pair of opposing moments per element:

$$\mathbf{f}_{\mathrm{eq}} = EI\kappa_e \cdot \begin{bmatrix} 0 \\ -1 \\ 0 \\ +1 \end{bmatrix}$$

This drives the beam upward (positive transverse deflection $w$).

---

## 3. Von Mises Plasticity (`physics/plasticity.py`)

### Von Mises Yield Criterion

Yielding occurs when the **deviatoric** part of the stress tensor reaches a critical value. The Von Mises stress in terms of principal stresses is:

$$\sigma_{\mathrm{vm}} = \sqrt{\frac{1}{2}\left[(\sigma_1-\sigma_2)^2 + (\sigma_2-\sigma_3)^2 + (\sigma_3-\sigma_1)^2\right]}$$

In terms of Cartesian components:

$$\sigma_{\mathrm{vm}} = \sqrt{\frac{1}{2}\left[(\sigma_{xx}-\sigma_{yy})^2 + (\sigma_{yy}-\sigma_{zz})^2 + (\sigma_{zz}-\sigma_{xx})^2 + 6\left(\sigma_{xy}^2+\sigma_{yz}^2+\sigma_{xz}^2\right)\right]}$$

Yielding occurs when $\sigma_{\mathrm{vm}} \geq \sigma_y$, where $\sigma_y = 250\ \mathrm{MPa}$ (aluminium default).  
For a 1-D beam bending state $(\sigma_2 = \sigma_3 = 0)$, this reduces to $\sigma_{\mathrm{vm}} = |\sigma_1|$.

### Yield Surface

In principal stress space the Von Mises criterion defines a **cylinder** whose axis is the hydrostatic line $\sigma_1 = \sigma_2 = \sigma_3$. The cylinder radius is:

$$r = \sigma_y \sqrt{\frac{2}{3}}$$

### The Pi-Plane

The **pi-plane** ($\pi$-plane) is perpendicular to the hydrostatic axis, passing through the origin ($\sigma_1 + \sigma_2 + \sigma_3 = 0$). Projected onto this plane the yield locus is a **circle** of radius $r = \sigma_y\sqrt{2/3}$.

### Volumetric–Deviatoric Split

Any stress state decomposes as:

$$\sigma_{ij} = \sigma_{\mathrm{vol}}\,\delta_{ij} + s_{ij}$$

where the hydrostatic part is $\sigma_{\mathrm{vol}} = \dfrac{1}{3}\,\mathrm{tr}(\sigma)$ and $s_{ij}$ is the deviatoric part.  
**Key insight**: hydrostatic pressure alone cannot cause yielding — only the deviatoric part drives plastic flow.

### Distortion Energy

The strain energy stored in shape change per unit volume (normalised by $G$):

$$\varphi_d \propto (\sigma_1-\sigma_2)^2 + (\sigma_2-\sigma_3)^2 + (\sigma_3-\sigma_1)^2$$

This is proportional to $\sigma_{\mathrm{vm}}^2$. The Von Mises criterion is equivalent to yielding when **distortion energy** reaches a critical value.

---

## 4. Finite Element Method

### 4a. Euler-Bernoulli Beam Element (`fem/beam_element.py`)

Euler-Bernoulli theory assumes plane cross-sections remain plane and perpendicular to the neutral axis; shear deformation is neglected. The kinematic strain-curvature relation is:

$$\varepsilon_{xx}(x, z) = -z \frac{d^2 w}{dx^2} = -z\,\kappa$$

Each element has 2 nodes with 2 DOFs each (transverse displacement $w$ and rotation $\theta = dw/dx$):

$$\mathbf{u}_e = \begin{bmatrix} w_i \\ \theta_i \\ w_j \\ \theta_j \end{bmatrix}$$

**Hermite shape functions** $N_1, N_2, N_3, N_4$ ensure $C^1$ continuity (displacement and slope both continuous across elements).

**Element stiffness matrix** (4 x 4):

$$\mathbf{K}_e = \frac{EI}{L_e^3}
\begin{bmatrix}
12    &  6L_e   & -12   &  6L_e  \\
6L_e  &  4L_e^2 & -6L_e &  2L_e^2 \\
-12   & -6L_e   &  12   & -6L_e  \\
6L_e  &  2L_e^2 & -6L_e &  4L_e^2
\end{bmatrix}$$

**Consistent mass matrix** (4 x 4), derived from kinetic energy $\frac{1}{2}\int \rho A \dot{w}^2\,dx$:

$$\mathbf{M}_e = \frac{\rho A L_e}{420}
\begin{bmatrix}
 156   &  22L_e   &  54   & -13L_e  \\
 22L_e &   4L_e^2 &  13L_e & -3L_e^2 \\
  54   &  13L_e   & 156   & -22L_e  \\
-13L_e &  -3L_e^2 & -22L_e &  4L_e^2
\end{bmatrix}$$

### 4b. Constant Strain Triangle (CST) Element (`fem/plate_element.py`)

The CST element is the simplest 2-D triangular finite element for plane-stress problems. Three nodes, 2 DOFs each $(u, v)$, giving a $6 \times 6$ stiffness matrix. The strain field is **constant** over each element:

$$\varepsilon = \mathbf{B}\,\mathbf{u}_e$$

The strain-displacement matrix $\mathbf{B}$ (3 x 6) is computed from nodal coordinates $(x_i, y_i)$:

$$\mathbf{B} = \frac{1}{2A}
\begin{bmatrix}
b_1 & 0   & b_2 & 0   & b_3 & 0   \\
0   & c_1 & 0   & c_2 & 0   & c_3 \\
c_1 & b_1 & c_2 & b_2 & c_3 & b_3
\end{bmatrix}$$

where $b_i = y_j - y_k$ and $c_i = x_k - x_j$ (cyclic permutations over nodes 1, 2, 3), and $A$ is the element area.

**Element stiffness**:

$$\mathbf{K}_e = t \cdot A \cdot \mathbf{B}^T \mathbf{D}\, \mathbf{B}$$

The rectangle is meshed by splitting each quadrilateral cell diagonally into two triangles.

### 4c. Global Assembly (`fem/assembly.py`)

Element matrices are **scatter-added** into the global $\mathbf{K}$ and $\mathbf{M}$ at the appropriate global DOF indices. Boundary conditions are applied by **partitioning**: rows/columns for fixed DOFs are removed, leaving the reduced system:

$$\mathbf{K}_{ff}\,\mathbf{u}_f = \mathbf{F}_f$$

### 4d. Solvers (`fem/solver.py`)

**Static solver**: solves $\mathbf{K}_{ff}\,\mathbf{u}_f = \mathbf{F}_f$ by direct LU decomposition (`numpy.linalg.solve`). Fixed DOFs are reconstructed as zeros.

**Modal solver**: solves the generalised eigenvalue problem

$$\mathbf{K}_{ff}\,\phi = \lambda\,\mathbf{M}_{ff}\,\phi$$

using `scipy.linalg.eigh`. Natural frequencies are extracted as:

$$f_n = \frac{\sqrt{\lambda_n}}{2\pi} \quad [\mathrm{Hz}]$$

**Stress recovery**: at each element midpoint the curvature is:

$$\kappa = \mathbf{B}(L_e/2)\,\mathbf{u}_e$$

The bending stress at the top fibre ($z = +h/2$) is:

$$\sigma_{xx} = E\left(\kappa \cdot \frac{h}{2} - \varepsilon_e\right)$$

The subtracted $\varepsilon_e$ ensures only the **elastic** (mechanical minus eigen) strain produces stress.

---

## 5. Simulations

### 5a. Cantilever Beam Bending (`simulations/beam_bending.py`)

**Setup**: Aluminium cantilever ($E = 70\ \mathrm{GPa}$, 100 mm long, 5 mm wide, 1 mm thick) clamped at $x = 0$, free at $x = L$, with a uniform electrostrictive actuator on its top surface.

**Physics computed**:
- Tip deflection profile for a given voltage
- Quadratic validation: $w_{\mathrm{tip}} \propto V^2$ (confirming the electrostrictive $\varepsilon \propto V^2$ law)
- Thickness sensitivity: thinner beams are more flexible but also give larger $\varepsilon_e = M(V/t)^2$, creating a competing effect
- Bending stress: $\sigma = E(\kappa h/2 - \varepsilon_e)$; maximum at the clamped root
- Von Mises field (equals $|\sigma_{xx}|$ for uniaxial bending)
- Electrostrictive energy vs. voltage
- **Mohr's circle** at the peak-stress element: shows principal directions and maximum shear
- Distortion energy along the beam

### 5b. Bimetal Strip (`simulations/bimetal.py`)

A two-layer composite beam (aluminium top / steel bottom) bends due to a **mismatch strain** analogous to differential thermal expansion — the classical Timoshenko bimetal problem.

**Neutral axis** of the composite (modulus-weighted centroid):

$$\bar{y} = \frac{E_1 A_1 (h_2 + h_1/2) + E_2 A_2 (h_2/2)}{E_1 A_1 + E_2 A_2}$$

**Effective bending stiffness** (parallel axis theorem about the neutral axis):

$$EI_{\mathrm{eff}} = E_1 I_1 + E_2 I_2$$

**Mismatch strain**: $\varepsilon^{*} = \alpha_{\Delta} \Delta T$ with $\alpha_{\Delta} = 12 \times 10^{-6}\ \mathrm{K^{-1}}$ and $\Delta T = 50\ \mathrm{K}$. This drives an equivalent bending moment:

$$M^{*} = \frac{E_1 A_1\, E_2 A_2}{E_1 A_1 + E_2 A_2} \cdot \varepsilon^{*} \cdot \frac{h_1+h_2}{2}$$

The simulation compares bimetal deflection with electrostrictive deflection at the same voltage.

### 5c. Sinusoidal Actuation (`simulations/sinusoidal_actuation.py`)

The actuator strain varies along the beam:

$$\varepsilon_e(x) = M \left(\frac{V}{t}\right)^2 \sin\left(\frac{\pi x}{L}\right)$$

This produces a non-uniform curvature distribution and a more complex deformed shape — relevant to segmented-electrode or gradient-field actuator designs.

### 5d. Boundary Condition Comparison (`simulations/boundary_effects.py`)

The same electrostrictive load is applied under three classical support configurations:

| Configuration | Fixed DOFs | Expected behaviour |
|---|---|---|
| **Cantilever** (clamped-free) | $w_0,\, \theta_0$ | Maximum tip deflection |
| **Clamped-clamped** | $w_0,\, \theta_0,\, w_L,\, \theta_L$ | Minimal deflection, large internal moments |
| **Simply-supported** | $w_0,\, w_L$ (transverse only) | Intermediate deflection |

### 5e. Modal Analysis (`simulations/modal_analysis.py`)

Solves the generalised eigenvalue problem $\mathbf{K}\phi = \lambda\mathbf{M}\phi$ for the first six free-vibration natural frequencies and mode shapes of the cantilever beam.

Analytical reference for a uniform Euler-Bernoulli cantilever:

$$f_n = \frac{(\beta_n L)^2}{2\pi L^2}\sqrt{\frac{EI}{\rho A}}$$

where $(\beta_n L)^2 = 3.516,\ 22.03,\ 61.70,\ldots$ for modes 1–3. FEM converges to these as the mesh is refined.

### 5f. Plastic Yielding (`simulations/plastic_yielding.py`)

Tracks where $\sigma_{\mathrm{vm}} \geq \sigma_y$ as voltage increases. Key demonstrations:

- **Plastic zone growth**: as $V$ increases, $\varepsilon_e \propto V^2$, so the zone near the clamped root yields first
- **3-D yield cylinder** in $(\sigma_1, \sigma_2, \sigma_3)$ space
- **Pi-plane projection**: stress states vs. the yield circle
- **Hydrostatic vs. Von Mises**: explicit proof that $\sigma_{\mathrm{hyd}} = (\sigma_1+\sigma_2+\sigma_3)/3$ causes no yielding
- **Stress evolution surface**: 3-D plot of bending stress as a function of both position $x$ and voltage $V$

### 5g. 2-D Plate FEM (`simulations/plate_2d.py`)

CST plane-stress elements on a $100 \times 50\ \mathrm{mm}$ plate clamped on its left edge. Electrostrictive eigen-strain $\varepsilon_{xx} = M(V/t)^2$ is converted to equivalent nodal forces per element:

$$\mathbf{f}_{\mathrm{eq}}^{(e)} = \mathbf{B}^T \mathbf{D}\,\varepsilon^{*} \cdot A_e \cdot t$$

Outputs: full 2-D displacement field $(u_x, u_y)$, Von Mises stress contours at nodes, and deformed-vs-undeformed mesh (magnification $\times 10^4$).

### 5h. Design Optimisation (`simulations/optimization.py`)

Sweeps the 2-D design space $(V,\, t)$ to maximise tip deflection subject to:

$$\sigma_{\mathrm{max}}(V, t) \leq \sigma_y$$

A **Pareto front** is extracted: for each thickness the highest feasible voltage is identified. Physical insight — thinner beams give larger $\varepsilon_e \propto 1/t^2$ but also larger stresses $\propto 1/t^3$; there is an optimal thickness that balances flexibility against yield.

---

## 6. Stress Utilities (`utils/stress_utils.py`)

| Function | Physics |
|---|---|
| `volumetric_deviatoric_3d` | Full 3-D split: $\sigma_{\mathrm{vol}} = \mathrm{tr}(\sigma)/3$, deviatoric vector |
| `beam_stress_along_depth` | Linear bending stress through thickness: $\sigma(z) = \sigma_{\mathrm{axial}} + \sigma_{\mathrm{bending}}(z/c)$ |
| `stress_mohr_circle_2d` | Mohr's circle centre $= (\sigma_{xx}+\sigma_{yy})/2$, radius $= \sqrt{((\sigma_{xx}-\sigma_{yy})/2)^2 + \sigma_{xy}^2}$ |
| `von_mises_beam` | $\sigma_{\mathrm{vm}} = \sqrt{\sigma^2 + 3\tau^2}$ for bending + shear |

---

## 7. Default Material and Geometry Parameters (`config.py`)

| Parameter | Value | Description |
|---|---|---|
| $E$ | 70 GPa | Young's modulus (aluminium) |
| $\nu$ | 0.33 | Poisson's ratio |
| $\rho$ | 2700 kg/m³ | Density |
| $\sigma_y$ | 250 MPa | Yield stress |
| $M$ | $10^{-18}$ m²/V² | Electrostrictive coefficient |
| $t$ | 1 mm | Actuator / beam thickness |
| $L$ | 100 mm | Beam length |
| $b$ | 5 mm | Beam width |
| $n_{\mathrm{elem}}$ | 40 | Number of beam elements |
| $V_{\mathrm{default}}$ | 100 V | Default operating voltage |
| $V_{\mathrm{max}}$ | 300 V | Maximum voltage for sweeps |
| $E_1$ (Al) / $E_2$ (Steel) | 70 / 200 GPa | Bimetal layer moduli |
| $\alpha_{\Delta}$ | $12 \times 10^{-6}$ K⁻¹ | CTE mismatch |
| $\Delta T$ | 50 K | Temperature load |

---

## Summary of Physics Interactions

```
Voltage V
   │
   ▼  eps_e = M(V/t)^2          [Electrostriction — quadratic in V]
Eigen-strain eps_e
   │
   ▼  f_eq = EI*kappa_e*[0,-1,0,+1]   [Virtual Work — equiv. nodal moment]
Equivalent Load F
   │
   ▼  K*u = F                   [FEM Static Equilibrium]
Displacement u
   │
   ├──► kappa = B*u             [Kinematics — curvature from B matrix]
   │       │
   │       ▼  sigma = E*(kappa*h/2 - eps_e)   [Elasticity — top-fibre stress]
   │    Stress sigma
   │       │
   │       ▼  sigma_vm = |sigma| >= sigma_y?   [Von Mises — yield check]
   │    Plastic zone
   │
   └──► Modal: K*phi = lambda*M*phi   [Dynamics — frequencies and mode shapes]
```

All simulations are driven by electrostrictive actuation. The FEM framework correctly accounts for the eigen-strain in both the **load vector** (via equivalent nodal moments) and the **stress recovery step** (subtracting $\varepsilon_e$ before computing elastic stress).
