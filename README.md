# Electrostrictive FEM + Plasticity Simulation Engine

A modular Python FEM framework for analysing electrostrictive actuation, structural deformation, and plastic yielding — built for the AE246 Structures Lab (IIT Bombay, Sem 4).

---

## Physics Overview

### Electrostrictive Strain
$$\varepsilon_{\text{electro}} = M \left(\frac{V}{t}\right)^2$$

### Total Strain & Stress
$$\varepsilon_{\text{total}} = \varepsilon_{\text{mechanical}} + \varepsilon_{\text{electro}}$$
$$\sigma = D\,(\varepsilon - \varepsilon_{\text{electro}})$$

### FEM Equation
$$K\,u = F_{\text{equivalent}}$$

### Von Mises Yield Criterion
$$\sigma_{\text{vm}} = \sqrt{\tfrac{1}{2}\left[(\sigma_1-\sigma_2)^2 + (\sigma_2-\sigma_3)^2 + (\sigma_3-\sigma_1)^2\right]} \geq \sigma_y$$

---

## Project Structure

```
FEM_code_matrix/
│
├── main.py                        # Master orchestrator — runs all/selected simulations
├── config.py                      # All material & geometry parameters
│
├── physics/
│   ├── electrostriction.py        # ε = M(V/t)², spatial/sinusoidal variant, energy
│   ├── elasticity.py              # D matrices (plane stress/strain), principal stresses
│   └── plasticity.py              # Von Mises, yield surface, π-plane, distortion energy
│
├── fem/
│   ├── beam_element.py            # Euler-Bernoulli Ke, Me, electrostrictive nodal forces
│   ├── assembly.py                # Global K/M/F assembly, BC partitioning
│   ├── solver.py                  # Static solver, modal eigensolver, stress recovery
│   └── plate_element.py           # CST element, rectangular mesh generator, plate assembly
│
├── simulations/
│   ├── beam_bending.py            # Cantilever under electrostrictive load (Basic)
│   ├── bimetal.py                 # Bimetal strip with E-mismatch strain (Intermediate)
│   ├── sinusoidal_actuation.py    # ε(x) = M(V/t)² sin(πx/L) — wave shapes (Intermediate)
│   ├── boundary_effects.py        # Cantilever / Clamped-Clamped / Simply-Supported (Intermediate)
│   ├── modal_analysis.py          # Kφ = λMφ — natural frequencies & mode shapes (Advanced)
│   ├── plastic_yielding.py        # Full plasticity analysis — yield zones & surfaces (Advanced)
│   ├── plate_2d.py                # 2-D CST plate FEM with electrostrictive body force (Advanced)
│   └── optimization.py            # Pareto front: max δ subject to no yielding (Advanced)
│
├── utils/
│   ├── stress_utils.py            # Mohr's circle, vol/dev split, stress through beam depth
│   └── math_utils.py              # General math helpers
│
├── visualization/
│   ├── plots.py                   # 20+ presentation-ready static plot functions
│   └── animations.py              # 4 GIF animation generators
│
├── results/
│   ├── plots/                     # All PNG outputs (auto-created)
│   └── animations/                # All GIF outputs (auto-created)
│
└── baseline.py                    # Original single-file prototype (reference only)
```

---

## Installation

```bash
pip install numpy scipy matplotlib pillow
```

> Python 3.9+ recommended. No other dependencies required.

---

## Running the Code

### Run Everything (all 8 simulations)

```bash
python main.py
```

### Run Individual Simulations

```bash
python main.py beam          # Cantilever beam bending + V-sweep + thickness study
python main.py bimetal       # Bimetal strip deflection
python main.py sinusoidal    # Sinusoidal actuation wave shapes
python main.py bc            # Boundary condition comparison
python main.py modal         # Modal analysis (natural frequencies + mode shapes)
python main.py plastic       # Plasticity — Von Mises, yield surface, π-plane
python main.py plate         # 2-D plate FEM
python main.py opt           # Optimisation — Pareto front
```

### Run Multiple Specific Simulations

```bash
python main.py beam modal plastic
python main.py plate opt
```

### Run Simulations Directly

```bash
python simulations/beam_bending.py
python simulations/modal_analysis.py
python simulations/plastic_yielding.py
python simulations/plate_2d.py
python simulations/optimization.py
python simulations/bimetal.py
python simulations/sinusoidal_actuation.py
python simulations/boundary_effects.py
```

---

## Configuration

All parameters are in `config.py`. Key values:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `E` | 70 GPa | Young's modulus (Aluminium) |
| `nu` | 0.33 | Poisson's ratio |
| `sigma_y` | 250 MPa | Yield stress |
| `M` | 1×10⁻¹⁸ m²/V² | Electrostrictive coefficient |
| `t` | 1 mm | Actuator thickness |
| `L` | 100 mm | Beam length |
| `V_default` | 100 V | Default operating voltage |
| `V_max` | 300 V | Max voltage for sweeps |
| `n_elem` | 40 | Number of beam elements |

---

## Outputs

All outputs are saved automatically to `results/`. The `results/` folder is created on first run.

### Static Plots (`results/plots/`)

| File | Description |
|------|-------------|
| `01_beam_deflection.png` | Cantilever deflection profile at V = 100 V |
| `02_voltage_vs_disp.png` | Tip displacement vs V² — validates ε ∝ V² linearity |
| `03_thickness_vs_disp.png` | Deflection sensitivity to actuator thickness |
| `04_stress_distribution.png` | Bending stress along beam length |
| `05_von_mises_beam.png` | Von Mises field + yield zone indicator |
| `06_energy_vs_V.png` | Electrostrictive stored energy vs voltage |
| `07_mohr_circle.png` | Mohr's circle at maximum stress location |
| `08_distortion_energy.png` | Distortion energy ratio — yield trigger visualisation |
| `09_bimetal_deflection.png` | Bimetal strip deflection from mismatch strain |
| `10_bimetal_vs_electro.png` | Bimetal vs electrostrictive actuation comparison |
| `11_sinusoidal_deformation.png` | Multi-voltage sinusoidal wave-shape overlay |
| `12_bc_comparison.png` | Three boundary conditions compared |
| `13_mode_shapes.png` | 6 mode shapes subplot grid |
| `14_freq_bar.png` | Natural frequency bar chart |
| `15_von_mises_high_V.png` | Von Mises field at max voltage (V = 300 V) |
| `16_stress_high_V.png` | Bending stress at max voltage |
| `17_yield_surface_3d.png` | Von Mises yield cylinder in (σ₁, σ₂, σ₃) space |
| `18_pi_plane.png` | π-plane circular yield locus with stress state points |
| `19_vol_vs_dev.png` | Volumetric vs deviatoric stress — deviatoric causes yielding |
| `20_distortion_energy.png` | Distortion energy field at high voltage |
| `21_plastic_zone_growth.png` | Plastic volume fraction vs voltage |
| `22_stress_surface.png` | 3-D surface: stress vs voltage × position |
| `23_hydrostatic_vs_vm.png` | Hydrostatic stress vs Von Mises — key plasticity insight |
| `24_plate_displacement.png` | 2-D plate x-displacement and magnitude field |
| `25_plate_stress.png` | 2-D plate Von Mises stress contour |
| `26_plate_mesh_deformed.png` | Deformed vs undeformed mesh overlay |
| `27_optimisation_landscape.png` | Deflection + stress contour landscape (V vs t) |
| `28_pareto_front.png` | Optimal (V, t) pairs maximising deflection within yield |

### Animations (`results/animations/`)

| File | Description |
|------|-------------|
| `01_beam_voltage_anim.gif` | Beam deflection growing as voltage sweeps 0 → 300 V |
| `02_sinusoidal_anim.gif` | Sinusoidal wave-shape evolution with voltage |
| `03_mode_shapes_anim.gif` | First 4 mode shapes oscillating in time |
| `04_plastic_growth_anim.gif` | Plastic zone spreading as voltage increases |

---

## Key Physics — Presentation Highlights

1. **ε ∝ V²** — Plot `02` shows the quadratic relationship, confirming electrostrictive physics (unlike piezoelectric which is linear in E)
2. **Mode shapes** — `13` shows 6 modes from 82 Hz to 6984 Hz, relevant for docking vibration
3. **Yield surface** — `17` is the full 3-D Von Mises cylinder; `18` is the π-plane circle
4. **Hydrostatic stress** — `23` demonstrates hydrostatic stress does NOT trigger yielding; only deviatoric stress matters
5. **Pareto front** — `28` shows optimal actuator design: thinner = more deflection per volt but higher stress risk
6. **Bimetal analogy** — `10` shows how thermal/mismatch strain and electrostrictive strain produce the same bending physics

---

## Module Reference

### `physics/electrostriction.py`
```python
electro_strain(V, t, M)                       # uniform ε = M(V/t)²
spatial_electro_strain(x, L, V, t, M)         # sinusoidal ε(x)
```

### `physics/plasticity.py`
```python
von_mises_from_principals(s1, s2, s3)         # σ_vm from principal stresses
von_mises_from_components(sxx, syy, ...)      # σ_vm from components
yield_surface_cylinder(sigma_y)               # 3-D cylinder (X, Y, Z arrays)
yield_locus_pi_plane(sigma_y)                 # π-plane circle (ξ, η)
distortion_energy(s1, s2, s3)                 # φ_distortion bracket term
```

### `fem/assembly.py`
```python
assemble_beam(n_elem, Le, E, I, rho, A, eps_e_array)  # returns K, M, F
apply_bc(K, F, fixed_dofs, M)                          # returns partitioned system
```

### `fem/solver.py`
```python
solve_static(K_ff, F_f, free_dofs, ndof)        # returns full u vector
solve_modal(K_ff, M_ff, n_modes)                # returns (freqs_hz, mode_vectors)
recover_beam_stress(u, n_elem, Le, E, I, h, eps_e_array)  # returns (x_mid, stress, kappa)
```
