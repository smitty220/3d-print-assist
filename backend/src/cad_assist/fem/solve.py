"""Linear static solve on a tet10 mesh: K u = f with some DOFs held at zero. CG preconditioned by smoothed-aggregation
AMG with the six rigid-body modes as near-nullspace (as the topology solver does)."""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import pyamg
import scipy.sparse.linalg as spla

from .element import elasticity, gauss_stresses, nodal_stresses, stiffness
from .mesh import Mesh


def rigid_modes(nodes: np.ndarray) -> np.ndarray:
    n = len(nodes)
    c = nodes - nodes.mean(axis=0)
    B = np.zeros((3 * n, 6))
    for k in range(3):
        B[k::3, k] = 1.0
    # rotations about x, y, z
    B[1::3, 3], B[2::3, 3] = -c[:, 2], c[:, 1]
    B[0::3, 4], B[2::3, 4] = c[:, 2], -c[:, 0]
    B[0::3, 5], B[1::3, 5] = -c[:, 1], c[:, 0]
    return B


@dataclass
class Solution:
    u: np.ndarray  # (3N,) mm
    stress: np.ndarray  # (N, 6) MPa, nodal
    gauss_stress: np.ndarray  # (M, 4, 6)
    reactions: np.ndarray  # (3N,) N, at held DOFs (K u - f)
    iterations: int
    seconds: dict


def solve(mesh: Mesh, E: float, nu: float, f: np.ndarray, fixed: np.ndarray, rtol: float = 1e-8, log=None, monitor=None) -> Solution:
    """`monitor(stage, **info)` follows the solve: "assemble", "precondition" (AMG levels), "iterate" after every CG
    iteration with the relative residual |f - K u| / |f| (the convergence measure, against `rtol`), then "stress"."""
    mon = monitor or (lambda *_a, **_k: None)
    t0 = time.time()
    mon("assemble", dof=mesh.n_dof)
    D = elasticity(E, nu)
    K = stiffness(mesh.nodes, mesh.tets, D)
    t1 = time.time()
    n = K.shape[0]
    free = np.setdiff1d(np.arange(n), fixed)
    Kf = K[free][:, free].tocsr()
    B = rigid_modes(mesh.nodes)[free]
    mon("precondition", seconds=t1 - t0)
    ml = pyamg.smoothed_aggregation_solver(Kf, B=B, symmetry="symmetric", max_coarse=600, coarse_solver="splu")
    mon("iterate", iteration=0, residual=1.0, rtol=rtol, levels=len(ml.levels))
    its = [0]
    b = f[free]
    bn = float(np.linalg.norm(b)) or 1.0

    def count(x):
        its[0] += 1
        if monitor is not None:  # one extra product with K per iteration (cheap beside the AMG cycle)
            mon("iterate", iteration=its[0], residual=float(np.linalg.norm(b - Kf @ x)) / bn, rtol=rtol)

    uf, info = spla.cg(Kf, b, rtol=rtol, maxiter=2000, M=ml.aspreconditioner(cycle="V"), callback=count)
    if info != 0:
        raise RuntimeError(f"The solver did not converge ({its[0]} iterations); check the supports hold every rigid-body motion")
    t2 = time.time()
    mon("stress", iterations=its[0])
    u = np.zeros(n)
    u[free] = uf
    sg = gauss_stresses(mesh.nodes, mesh.tets, D, u)
    stress = nodal_stresses(mesh.tets, len(mesh.nodes), sg)
    t3 = time.time()
    if log:
        log(f"solved {n:,} DOF in {its[0]} iterations")
    return Solution(u, stress, sg, K @ u - f, its[0], {"assemble": t1 - t0, "solve": t2 - t1, "stress": t3 - t2})
