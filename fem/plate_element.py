"""
fem/plate_element.py
Constant Strain Triangle (CST) element for 2-D plane-stress plate FEM.
"""
import numpy as np


def cst_stiffness(xy, D, thickness=1.0):
    """
    Compute 6×6 stiffness matrix for a CST element.

    Parameters
    ----------
    xy        : (3,2) nodal coordinates [[x1,y1],[x2,y2],[x3,y3]]
    D         : (3,3) plane-stress constitutive matrix
    thickness : element thickness

    Returns
    -------
    Ke : (6,6) element stiffness matrix
    """
    x1, y1 = xy[0]
    x2, y2 = xy[1]
    x3, y3 = xy[2]

    A2 = (x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1)  # 2*Area
    A  = abs(A2) / 2.0

    if A < 1e-30:
        return np.zeros((6, 6))

    # Strain-displacement matrix B (3×6), constant over element
    b1 = y2 - y3;  b2 = y3 - y1;  b3 = y1 - y2
    c1 = x3 - x2;  c2 = x1 - x3;  c3 = x2 - x1

    B = (1.0 / (2 * A)) * np.array([
        [b1,  0, b2,  0, b3,  0],
        [ 0, c1,  0, c2,  0, c3],
        [c1, b1, c2, b2, c3, b3]
    ])

    Ke = thickness * A * (B.T @ D @ B)
    return Ke, B, A


def cst_stress(u_e, B, D, eps_eigen=None):
    """
    Recover stress from element displacement vector u_e (6,).
    """
    if eps_eigen is None:
        eps_eigen = np.zeros(3)
    eps = B @ u_e
    sigma = D @ (eps - eps_eigen)
    return sigma, eps


def mesh_rect_cst(Lx, Ly, nx, ny):
    """
    Create a structured CST mesh over a rectangle [0,Lx]×[0,Ly].
    Each quad is split into 2 triangles.

    Returns
    -------
    nodes    : (n_nodes, 2) array of (x, y)
    elements : (n_elem, 3) connectivity (node indices)
    """
    xs = np.linspace(0, Lx, nx + 1)
    ys = np.linspace(0, Ly, ny + 1)
    X, Y = np.meshgrid(xs, ys)
    nodes = np.column_stack([X.ravel(), Y.ravel()])

    def nid(i, j):               # row-major node index
        return j * (nx + 1) + i

    elements = []
    for j in range(ny):
        for i in range(nx):
            n0 = nid(i,   j)
            n1 = nid(i+1, j)
            n2 = nid(i+1, j+1)
            n3 = nid(i,   j+1)
            elements.append([n0, n1, n2])   # lower triangle
            elements.append([n0, n2, n3])   # upper triangle

    return nodes, np.array(elements)


def assemble_plate(nodes, elements, D, thickness=1.0):
    """
    Assemble global stiffness matrix for CST plate FEM.
    DOFs: 2 per node (u, v) — plane-stress displacements.

    Returns
    -------
    K : (2*n_nodes, 2*n_nodes)
    Bs : list of B matrices per element
    As : list of areas per element
    """
    n_nodes = nodes.shape[0]
    ndof = 2 * n_nodes
    K = np.zeros((ndof, ndof))
    Bs = []
    As = []

    for elem in elements:
        xy = nodes[elem]
        result = cst_stiffness(xy, D, thickness)
        if isinstance(result, tuple):
            Ke, B, A = result
        else:
            Bs.append(None)
            As.append(0.0)
            continue

        Bs.append(B)
        As.append(A)

        dofs = []
        for n in elem:
            dofs += [2*n, 2*n+1]
        dofs = np.array(dofs)

        for i in range(6):
            for j in range(6):
                K[dofs[i], dofs[j]] += Ke[i, j]

    return K, Bs, As
