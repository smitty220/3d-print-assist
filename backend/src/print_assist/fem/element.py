"""10-node quadratic tetrahedra (C3D10), isoparametric, for linear-elastic stress analysis of the parametric part.

Node order is Gmsh's: corners 0-3, then mid-edge nodes 4:(0,1) 5:(1,2) 6:(0,2) 7:(0,3) 8:(2,3) 9:(1,3). Stiffness uses the
4-point Gauss rule (exact for straight-sided elements; as Abaqus integrates C3D10). Stresses are computed at the Gauss
points and extrapolated to the corners element by element, then averaged at nodes. Voigt order: xx, yy, zz, yz, xz, xy
(engineering shear strains).
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp

EDGES = np.array([(0, 1), (1, 2), (0, 2), (0, 3), (2, 3), (1, 3)])  # mid-edge nodes 4..9

# 4-point rule on the reference tetrahedron (volume 1/6).
_A, _B = 0.5854101966249685, 0.1381966011250105
GAUSS = np.array([(_B, _B, _B), (_A, _B, _B), (_B, _A, _B), (_B, _B, _A)])  # (xi, eta, zeta) = (L1, L2, L3)
WEIGHTS = np.full(4, 1 / 24)

_DL = np.array([(-1.0, -1.0, -1.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)])  # dL_i / d(xi, eta, zeta)


def _volume_coords(p: np.ndarray) -> np.ndarray:
    xi, eta, zeta = p
    return np.array([1 - xi - eta - zeta, xi, eta, zeta])


def shape(p: np.ndarray) -> np.ndarray:
    """N (10,) at reference point p = (xi, eta, zeta)."""
    L = _volume_coords(p)
    N = np.empty(10)
    N[:4] = L * (2 * L - 1)
    for k, (i, j) in enumerate(EDGES):
        N[4 + k] = 4 * L[i] * L[j]
    return N


def shape_grad(p: np.ndarray) -> np.ndarray:
    """dN/d(xi, eta, zeta) (10, 3) at p."""
    L = _volume_coords(p)
    dN = np.empty((10, 3))
    for i in range(4):
        dN[i] = (4 * L[i] - 1) * _DL[i]
    for k, (i, j) in enumerate(EDGES):
        dN[4 + k] = 4 * (L[j] * _DL[i] + L[i] * _DL[j])
    return dN


DN_GAUSS = np.stack([shape_grad(g) for g in GAUSS])  # (4, 10, 3)
# Extrapolation from the 4 Gauss points to the 4 corners with the linear field the points define.
_LIN = np.stack([_volume_coords(g) for g in GAUSS])  # (gauss, corner)
EXTRAP = np.linalg.inv(_LIN)  # corner values = EXTRAP @ gauss values


def elasticity(E: float, nu: float) -> np.ndarray:
    lam = E * nu / ((1 + nu) * (1 - 2 * nu))
    mu = E / (2 * (1 + nu))
    D = np.zeros((6, 6))
    D[:3, :3] = lam
    D[np.arange(3), np.arange(3)] += 2 * mu
    D[3:, 3:] = np.eye(3) * mu
    return D


def _b_matrices(coords: np.ndarray, g: int) -> tuple[np.ndarray, np.ndarray]:
    """B (m, 6, 30) and det J (m,) at Gauss point g for elements with node coordinates `coords` (m, 10, 3)."""
    dN = DN_GAUSS[g]  # (10, 3)
    J = np.einsum("mni,nj->mij", coords, dN)  # dx_i / dxi_j
    det = np.linalg.det(J)
    invJ = np.linalg.inv(J)
    dx = np.einsum("nj,mji->mni", dN, invJ)  # dN/dx (m, 10, 3)
    m = coords.shape[0]
    B = np.zeros((m, 6, 30))
    B[:, 0, 0::3] = dx[:, :, 0]
    B[:, 1, 1::3] = dx[:, :, 1]
    B[:, 2, 2::3] = dx[:, :, 2]
    B[:, 3, 1::3] = dx[:, :, 2]
    B[:, 3, 2::3] = dx[:, :, 1]
    B[:, 4, 0::3] = dx[:, :, 2]
    B[:, 4, 2::3] = dx[:, :, 0]
    B[:, 5, 0::3] = dx[:, :, 1]
    B[:, 5, 1::3] = dx[:, :, 0]
    return B, det


def element_dofs(tets: np.ndarray) -> np.ndarray:
    return (3 * tets[:, :, None] + np.arange(3)).reshape(len(tets), 30)


def stiffness(nodes: np.ndarray, tets: np.ndarray, D: np.ndarray, chunk: int = 8000) -> sp.csr_matrix:
    """Global stiffness (3N x 3N). Raises if an element is inverted (negative Jacobian)."""
    n = 3 * len(nodes)
    K = sp.csr_matrix((n, n))
    for s in range(0, len(tets), chunk):
        t = tets[s:s + chunk]
        coords = nodes[t]
        Ke = np.zeros((len(t), 30, 30))
        for g in range(4):
            B, det = _b_matrices(coords, g)
            if (det <= 0).any():
                raise ValueError(f"{int((det <= 0).sum())} inverted element(s) in the mesh")
            Ke += np.einsum("mki,kl,mlj,m->mij", B, D, B, det * WEIGHTS[g], optimize=True)
        dofs = element_dofs(t)
        rows = np.repeat(dofs, 30, axis=1).ravel()
        cols = np.tile(dofs, 30).ravel()
        K = K + sp.coo_matrix((Ke.ravel(), (rows, cols)), shape=(n, n)).tocsr()
    K.sum_duplicates()
    return K


def gauss_stresses(nodes: np.ndarray, tets: np.ndarray, D: np.ndarray, u: np.ndarray, chunk: int = 20000) -> np.ndarray:
    """Stress (m, 4, 6) at each element's Gauss points."""
    out = np.empty((len(tets), 4, 6))
    U = u.reshape(-1, 3)
    for s in range(0, len(tets), chunk):
        t = tets[s:s + chunk]
        coords = nodes[t]
        ue = U[t].reshape(len(t), 30)
        for g in range(4):
            B, _det = _b_matrices(coords, g)
            out[s:s + chunk, g] = np.einsum("kl,mlj,mj->mk", D, B, ue, optimize=True)
    return out


def nodal_stresses(tets: np.ndarray, n_nodes: int, sg: np.ndarray) -> np.ndarray:
    """Corner-node stresses (n_nodes, 6): Gauss values extrapolated per element, averaged over the elements at each node.
    Mid-edge nodes take the mean of their edge's corners."""
    corner = np.einsum("cg,mgk->mck", EXTRAP, sg)  # (m, 4, 6)
    acc = np.zeros((n_nodes, 6))
    cnt = np.zeros(n_nodes)
    np.add.at(acc, tets[:, :4].ravel(), corner.reshape(-1, 6))
    np.add.at(cnt, tets[:, :4].ravel(), 1)
    for k, (i, j) in enumerate(EDGES):
        mids = tets[:, 4 + k]
        np.add.at(acc, mids, 0.5 * (corner[:, i] + corner[:, j]))
        np.add.at(cnt, mids, 1)
    return acc / np.maximum(cnt, 1)[:, None]


def von_mises(s: np.ndarray) -> np.ndarray:
    xx, yy, zz, yz, xz, xy = s.T
    return np.sqrt(0.5 * ((xx - yy) ** 2 + (yy - zz) ** 2 + (zz - xx) ** 2) + 3 * (yz**2 + xz**2 + xy**2))


def principal(s: np.ndarray) -> np.ndarray:
    """Principal stresses (n, 3), largest first."""
    xx, yy, zz, yz, xz, xy = s.T
    T = np.stack([np.stack([xx, xy, xz], -1), np.stack([xy, yy, yz], -1), np.stack([xz, yz, zz], -1)], -2)
    return np.linalg.eigvalsh(T)[:, ::-1]


# ---------------------------------------------------------------- surface tractions (6-node triangles)

_TRI_EDGES = np.array([(0, 1), (1, 2), (0, 2)])  # Gmsh 6-node triangle: mid-edge nodes 3:(0,1) 4:(1,2) 5:(0,2)
# 6-point rule (degree 4) on the reference triangle (area 1/2).
_TA, _TB = 0.445948490915965, 0.091576213509771
_TW1, _TW2 = 0.223381589678011 / 2, 0.109951743655322 / 2
TRI_GAUSS = np.array([(_TA, _TA), (1 - 2 * _TA, _TA), (_TA, 1 - 2 * _TA), (_TB, _TB), (1 - 2 * _TB, _TB), (_TB, 1 - 2 * _TB)])
TRI_WEIGHTS = np.array([_TW1] * 3 + [_TW2] * 3)


def _tri_shape(p) -> tuple[np.ndarray, np.ndarray]:
    s, t = p
    L = np.array([1 - s - t, s, t])
    dL = np.array([(-1.0, -1.0), (1.0, 0.0), (0.0, 1.0)])
    N = np.empty(6)
    dN = np.empty((6, 2))
    N[:3] = L * (2 * L - 1)
    for i in range(3):
        dN[i] = (4 * L[i] - 1) * dL[i]
    for k, (i, j) in enumerate(_TRI_EDGES):
        N[3 + k] = 4 * L[i] * L[j]
        dN[3 + k] = 4 * (L[j] * dL[i] + L[i] * dL[j])
    return N, dN


_TRI_N, _TRI_DN = zip(*(_tri_shape(g) for g in TRI_GAUSS))
TRI_N = np.stack(_TRI_N)  # (6 gauss, 6 nodes)
TRI_DN = np.stack(_TRI_DN)  # (6 gauss, 6 nodes, 2)


def traction_load(nodes: np.ndarray, tris: np.ndarray, traction, n_nodes: int | None = None) -> np.ndarray:
    """Consistent nodal forces (3N,) for a traction t(x) (callable on (k, 3) points -> (k, 3)) over 6-node triangles."""
    n_nodes = n_nodes or len(nodes)
    f = np.zeros((n_nodes, 3))
    coords = nodes[tris]  # (F, 6, 3)
    for g in range(len(TRI_GAUSS)):
        x = np.einsum("n,fni->fi", TRI_N[g], coords)
        dxs = np.einsum("n,fni->fi", TRI_DN[g][:, 0], coords)
        dxt = np.einsum("n,fni->fi", TRI_DN[g][:, 1], coords)
        dA = np.linalg.norm(np.cross(dxs, dxt), axis=1) * TRI_WEIGHTS[g]
        tr = traction(x) * dA[:, None]  # (F, 3)
        np.add.at(f, tris.ravel(), (TRI_N[g][None, :, None] * tr[:, None, :]).reshape(-1, 3))
    return f.ravel()


def surface_points(nodes: np.ndarray, tris: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Gauss points (F*6, 3), outward-agnostic unit normals and area weights on 6-node triangles."""
    coords = nodes[tris]
    xs, ns, ws = [], [], []
    for g in range(len(TRI_GAUSS)):
        x = np.einsum("n,fni->fi", TRI_N[g], coords)
        dxs = np.einsum("n,fni->fi", TRI_DN[g][:, 0], coords)
        dxt = np.einsum("n,fni->fi", TRI_DN[g][:, 1], coords)
        c = np.cross(dxs, dxt)
        a = np.linalg.norm(c, axis=1)
        xs.append(x)
        ns.append(c / a[:, None])
        ws.append(a * TRI_WEIGHTS[g])
    return np.concatenate(xs), np.concatenate(ns), np.concatenate(ws)
