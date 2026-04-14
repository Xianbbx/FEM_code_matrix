import numpy as np
import matplotlib.pyplot as plt

# -------------------------
# PARAMETERS
# -------------------------
L = 0.1          # beam length (m)
E = 70e9         # Young's modulus (Al)
I = 1e-10        # moment of inertia
M = 1e-18        # electrostrictive coefficient
V = 100          # applied voltage
t = 1e-3         # thickness

n_elem = 20
n_nodes = n_elem + 1
dof = 2 * n_nodes

Le = L / n_elem

# Electrostrictive strain
eps_e = M * (V / t)**2

# -------------------------
# ELEMENT STIFFNESS
# -------------------------
def beam_element_stiffness(E, I, L):
    return (E * I / L**3) * np.array([
        [12, 6*L, -12, 6*L],
        [6*L, 4*L**2, -6*L, 2*L**2],
        [-12, -6*L, 12, -6*L],
        [6*L, 2*L**2, -6*L, 4*L**2]
    ])

K = np.zeros((dof, dof))
F = np.zeros(dof)

# -------------------------
# ASSEMBLY
# -------------------------
for e in range(n_elem):
    ke = beam_element_stiffness(E, I, Le)

    # DOF mapping
    idx = [2*e, 2*e+1, 2*e+2, 2*e+3]

    for i in range(4):
        for j in range(4):
            K[idx[i], idx[j]] += ke[i, j]

    # Electrostrictive equivalent force (approx)
    f_e = eps_e * E * I / Le * np.array([1, 0, -1, 0])
    for i in range(4):
        F[idx[i]] += f_e[i]

# -------------------------
# BOUNDARY CONDITIONS
# -------------------------
# Fixed left end
fixed_dofs = [0, 1]

free_dofs = list(set(range(dof)) - set(fixed_dofs))

K_ff = K[np.ix_(free_dofs, free_dofs)]
F_f = F[free_dofs]

# Solve
U = np.zeros(dof)
U[free_dofs] = np.linalg.solve(K_ff, F_f)

# -------------------------
# PLOTTING
# -------------------------
x = np.linspace(0, L, n_nodes)
w = U[0::2]

plt.plot(x, w * 1e6)
plt.xlabel("Length (m)")
plt.ylabel("Deflection (microns)")
plt.title("Electrostrictive Beam Bending")
plt.grid()
plt.show()

voltages = np.linspace(0, 150, 20)
max_disp = []

for V in voltages:
    eps_e = M * (V/t)**2
    # recompute FEM
    max_disp.append(max(w))

plt.plot(voltages, max_disp)
plt.title("Displacement vs Voltage^2")