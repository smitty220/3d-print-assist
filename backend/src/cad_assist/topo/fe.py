"""Linear elastic finite elements on a regular voxel grid (8-node hexahedra), for topology optimisation.

Element (i, j, k) spans [i, i+1]·h × [j, j+1]·h × [k, k+1]·h in grid coordinates; arrays over elements
are shaped (nx, ny, nz). Node (i, j, k) has index i + (nx+1)·(j + (ny+1)·k) and DOFs 3n, 3n+1, 3n+2.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pyamg
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from pyamg.multilevel import coarse_grid_solver
from pyamg.relaxation.smoothing import change_smoothers

# Local corner order: corner (a, b, c) ∈ {0,1}³ has local index a + 2b + 4c.
CORNERS = np.array([(a, b, c) for c in (0, 1) for b in (0, 1) for a in (0, 1)])


def hex8_stiffness(h: float, E: float = 1.0, nu: float = 0.33) -> np.ndarray:
    """24×24 stiffness of a cube of side h (2×2×2 Gauss), isotropic material."""
    lam = E / ((1 + nu) * (1 - 2 * nu))
    D = lam * np.array([
        [1 - nu, nu, nu, 0, 0, 0],
        [nu, 1 - nu, nu, 0, 0, 0],
        [nu, nu, 1 - nu, 0, 0, 0],
        [0, 0, 0, (1 - 2 * nu) / 2, 0, 0],
        [0, 0, 0, 0, (1 - 2 * nu) / 2, 0],
        [0, 0, 0, 0, 0, (1 - 2 * nu) / 2],
    ])
    s = CORNERS * 2 - 1
    g = 1 / np.sqrt(3)
    K = np.zeros((24, 24))
    for gp in np.array([(a, b, c) for a in (-g, g) for b in (-g, g) for c in (-g, g)]):
        dN = np.stack([
            s[:, 0] * (1 + s[:, 1] * gp[1]) * (1 + s[:, 2] * gp[2]),
            s[:, 1] * (1 + s[:, 0] * gp[0]) * (1 + s[:, 2] * gp[2]),
            s[:, 2] * (1 + s[:, 0] * gp[0]) * (1 + s[:, 1] * gp[1]),
        ], axis=1) / 8 * (2 / h)
        B = np.zeros((6, 24))
        B[0, 0::3], B[1, 1::3], B[2, 2::3] = dN[:, 0], dN[:, 1], dN[:, 2]
        B[3, 0::3], B[3, 1::3] = dN[:, 1], dN[:, 0]
        B[4, 1::3], B[4, 2::3] = dN[:, 2], dN[:, 1]
        B[5, 0::3], B[5, 2::3] = dN[:, 2], dN[:, 0]
        K += B.T @ D @ B * (h / 2) ** 3
    return K


@dataclass
class Grid:
    nx: int
    ny: int
    nz: int
    h: float  # element size, mm

    @property
    def shape(self) -> tuple[int, int, int]:
        return (self.nx, self.ny, self.nz)

    @property
    def n_nodes(self) -> int:
        return (self.nx + 1) * (self.ny + 1) * (self.nz + 1)

    def node(self, i, j, k):
        return i + (self.nx + 1) * (j + (self.ny + 1) * k)

    def element_nodes(self) -> np.ndarray:
        """(n_elements, 8) node indices, elements in C order of the (nx, ny, nz) array."""
        i, j, k = np.meshgrid(np.arange(self.nx), np.arange(self.ny), np.arange(self.nz), indexing="ij")
        i, j, k = i.ravel(), j.ravel(), k.ravel()
        return self.node(i[:, None] + CORNERS[:, 0], j[:, None] + CORNERS[:, 1], k[:, None] + CORNERS[:, 2])

    def node_coords(self) -> np.ndarray:
        """(n_nodes, 3) node positions in grid coordinates (mm from the grid origin)."""
        k, j, i = np.meshgrid(np.arange(self.nz + 1), np.arange(self.ny + 1), np.arange(self.nx + 1), indexing="ij")
        return np.stack([i.ravel(), j.ravel(), k.ravel()], axis=1) * self.h


@dataclass
class Springs:
    """Constant axial springs added to K: bearing contact at pin supports. A spring joins node `a` to node `b` (or to
    ground, b = -1) along unit direction `d` (grid axes) with stiffness `k`. Node indices past the grid's are extra
    nodes at `extra` (grid coordinates, mm): a pin centre that a link support holds."""

    extra: np.ndarray  # (m, 3)
    a: np.ndarray  # (s,) int
    b: np.ndarray  # (s,) int, -1 = ground
    d: np.ndarray  # (s, 3)
    k: np.ndarray  # (s,)

    def entries(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(row dof, col dof, value) of every spring entry, full numbering."""
        dd = self.k[:, None, None] * self.d[:, :, None] * self.d[:, None, :]  # (s, 3, 3)
        c = np.arange(3)
        rows, cols, vals = [], [], []
        for p, q, sign in ((self.a, self.a, 1), (self.b, self.b, 1), (self.a, self.b, -1), (self.b, self.a, -1)):
            use = (p >= 0) & (q >= 0)
            r, cc = np.broadcast_arrays(3 * p[use, None, None] + c[None, :, None], 3 * q[use, None, None] + c[None, None, :])
            rows.append(r.ravel())
            cols.append(cc.ravel())
            vals.append(sign * dd[use].ravel())
        return np.concatenate(rows), np.concatenate(cols), np.concatenate(vals)


class Stiffness:
    """Assembles K(E) for the elements in `active` with the DOFs in `fixed` removed, plus any constant `springs`.

    The sparsity pattern and each element entry's slot in it are worked out once, so a reassembly for new
    element moduli is a handful of vector adds. Nodes touched by no active element (or spring) are dropped too.
    """

    def __init__(self, grid: Grid, active: np.ndarray, fixed_dofs: np.ndarray, nu: float = 0.33, springs: Springs | None = None):
        self.grid = grid
        self.KE = hex8_stiffness(grid.h, 1.0, nu)
        self.elements = np.flatnonzero(active.ravel())
        nodes = grid.element_nodes()[self.elements]
        edof = (3 * nodes[:, :, None] + np.arange(3)).reshape(len(self.elements), 24)
        n_extra = len(springs.extra) if springs is not None else 0
        self.n_full = 3 * (grid.n_nodes + n_extra)
        sr, sc, sv = springs.entries() if springs is not None else (np.zeros(0, np.int64),) * 2 + (np.zeros(0),)
        keep = np.zeros(self.n_full, bool)
        keep[edof.ravel()] = True
        keep[sr] = True
        keep[fixed_dofs] = False
        self.free = np.flatnonzero(keep)
        dof_map = np.full(self.n_full, -1, np.int64)
        dof_map[self.free] = np.arange(len(self.free))
        self.edof = edof
        loc = dof_map[edof]  # (ne, 24) reduced dof or -1
        n = len(self.free)
        rows = np.repeat(loc, 24, axis=1).ravel()
        cols = np.tile(loc, 24).ravel()
        ok = (rows >= 0) & (cols >= 0)
        s_rows, s_cols = dof_map[sr], dof_map[sc]
        s_ok = (s_rows >= 0) & (s_cols >= 0)
        pattern = sp.coo_matrix((np.ones(ok.sum() + s_ok.sum(), np.int8),
                                 (np.concatenate([rows[ok], s_rows[s_ok]]), np.concatenate([cols[ok], s_cols[s_ok]]))), shape=(n, n)).tocsr()
        pattern.sum_duplicates()
        pattern.sort_indices()
        self.indptr, self.indices = pattern.indptr, pattern.indices
        # Slot of every (element, a, b) entry in the CSR data; -1 entries go to a dump slot at the end.
        key_csr = np.repeat(np.arange(n, dtype=np.int64), np.diff(self.indptr)) * n + self.indices
        key = rows * n + cols
        slot = np.searchsorted(key_csr, key)
        slot[~ok] = len(key_csr)
        self.slot = slot.reshape(len(self.elements), 576).T.astype(np.int32).copy()  # (576, ne)
        self.n = n
        self.nnz = len(key_csr)
        self.base = np.zeros(self.nnz + 1)  # the springs, in CSR data order
        np.add.at(self.base, np.searchsorted(key_csr, s_rows[s_ok] * n + s_cols[s_ok]), sv[s_ok])
        # Near-nullspace for smoothed aggregation: the three rigid translations and three rotations.
        coords = grid.node_coords() if not n_extra else np.vstack([grid.node_coords(), springs.extra])
        xyz = coords[self.free // 3]
        comp = self.free % 3
        B = np.zeros((n, 6))
        B[np.arange(n), comp] = 1
        c = xyz - xyz.mean(axis=0)
        # rotation about x: (0, -z, y); about y: (z, 0, -x); about z: (-y, x, 0)
        B[:, 3] = np.where(comp == 1, -c[:, 2], np.where(comp == 2, c[:, 1], 0))
        B[:, 4] = np.where(comp == 0, c[:, 2], np.where(comp == 2, -c[:, 0], 0))
        B[:, 5] = np.where(comp == 0, -c[:, 1], np.where(comp == 1, c[:, 0], 0))
        self.B = B

    def assemble(self, E_active: np.ndarray) -> sp.csr_matrix:
        """K for moduli E (one per active element, in `self.elements` order)."""
        data = self.base.copy()
        ke = self.KE.ravel()
        for ab in range(576):
            data[self.slot[ab]] += ke[ab] * E_active  # one element per slot for a fixed (a, b): no clashes
        return sp.csr_matrix((data[:-1], self.indices, self.indptr), shape=(self.n, self.n))

    def reduce(self, f_full: np.ndarray) -> np.ndarray:
        """Free entries of a force vector over the grid's nodes (extra nodes carry no load)."""
        return np.pad(f_full, (0, self.n_full - len(f_full)))[self.free]

    def expand(self, u_free: np.ndarray) -> np.ndarray:
        """Displacements over every node, grid nodes first, then any extra nodes."""
        u = np.zeros(self.n_full)
        u[self.free] = u_free
        return u


SMOOTHER = ("block_gauss_seidel", {"sweep": "symmetric"})  # pyamg's smoothed-aggregation default
# pyamg counts the coarsest-level limit in blocks (6 unknowns each with 6 rigid-body modes): 200 keeps the direct
# solve under ~1,200 unknowns. At 1500 a 20k-element mesh stopped at 6,000 and spent 0.7 s per factorisation.
MAX_COARSE_BLOCKS = 200


class AMG:
    """Smoothed-aggregation AMG preconditioner that is cheap to refresh as the densities change.

    A full build (strength of connection, aggregation, prolongation smoothing, Galerkin products) costs more than
    the CG solve itself. Between optimisation iterations the sparsity is fixed and only the moduli move, so the
    aggregates and prolongators are kept and only the coarse operators R·A·P are recomputed, with a full
    rebuild every `rebuild_every` updates. On the clevis this changed no CG iteration count.
    """

    def __init__(self, B: np.ndarray, rebuild_every: int = 8):
        self.B = B
        self.rebuild_every = rebuild_every
        self.ml = None
        self.updates = 0

    def update(self, K: sp.csr_matrix):
        if self.ml is None or self.updates % self.rebuild_every == 0:
            self.ml = pyamg.smoothed_aggregation_solver(K, B=self.B, max_coarse=MAX_COARSE_BLOCKS, symmetry="symmetric", coarse_solver="splu",
                                                        presmoother=SMOOTHER, postsmoother=SMOOTHER)
        else:
            levels = self.ml.levels
            levels[0].A = K
            for fine, coarse in zip(levels[:-1], levels[1:]):
                A = fine.R @ fine.A @ fine.P
                coarse.A = A.tobsr(blocksize=coarse.A.blocksize) if coarse.A.format == "bsr" else A.tocsr()
            change_smoothers(self.ml, SMOOTHER, SMOOTHER)  # block Gauss-Seidel caches inverted diagonal blocks
            self.ml.coarse_solver = coarse_grid_solver("splu")  # drops the cached factorisation
        self.updates += 1
        return self.ml


def solve(K: sp.csr_matrix, B: np.ndarray, rhs: list[np.ndarray], x0: list[np.ndarray | None] | None = None,
          rtol: float = 1e-6, maxiter: int = 400, cycle: str = "V", amg: AMG | None = None) -> list[np.ndarray]:
    """Solve K u = f for each right-hand side: CG preconditioned by smoothed-aggregation AMG (built once per call,
    or refreshed from `amg` when given)."""
    ml = (amg or AMG(B)).update(K)
    M = ml.aspreconditioner(cycle=cycle)
    out = []
    for i, f in enumerate(rhs):
        guess = x0[i] if x0 is not None else None
        u, info = spla.cg(K, f, x0=guess, rtol=rtol, maxiter=maxiter, M=M)
        out.append(u)
    return out
