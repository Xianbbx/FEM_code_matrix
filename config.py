# =============================================================
# config.py  —  All simulation parameters
# =============================================================
import numpy as np

# ---- Material (Aluminium default) ---------------------------
E        = 70e9          # Young's modulus [Pa]
nu       = 0.33          # Poisson's ratio
rho      = 2700.0        # density [kg/m³]
sigma_y  = 250e6         # yield stress [Pa]

# ---- Electrostrictive material parameters -------------------
M        = 1e-18         # electrostrictive coefficient [m²/V²]
t        = 1e-3          # actuator thickness [m]

# ---- Beam geometry ------------------------------------------
L        = 0.1           # beam length [m]
b        = 5e-3          # beam width [m]
h        = t             # beam height = thickness
I        = b * h**3 / 12 # second moment of area [m⁴]
A        = b * h         # cross-sectional area [m²]

# ---- Mesh ---------------------------------------------------
n_elem   = 40            # number of beam elements

# ---- Default voltage ----------------------------------------
V_default = 100.0        # [V]
V_max     = 300.0        # max voltage for sweeps [V]

# ---- Plate geometry (2-D) -----------------------------------
Lx = 0.1   # [m]
Ly = 0.05  # [m]
nx = 10    # divisions in x
ny = 5     # divisions in y

# ---- Modal analysis -----------------------------------------
n_modes  = 6

# ---- Bimetal layers -----------------------------------------
E_layer1 = 70e9    # Al
E_layer2 = 200e9   # Steel
alpha_mismatch = 12e-6   # CTE mismatch analogue [1/K]
delta_T = 0.001           # temperature-like load

# ---- Output directories ------------------------------------
RESULTS_DIR    = "results"
PLOTS_DIR      = "results/plots"
ANIM_DIR       = "results/animations"
