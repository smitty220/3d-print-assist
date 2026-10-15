"""Minimum-compliance topology optimisation (SIMP with optimality-criteria updates).

Maximises stiffness for a given amount of material: every element carries a density x in [0, 1],
stiffness E = Emin + x^p·(1 − Emin), and each iteration solves every load case, works out how much each
element adds to the (normalised, summed) stiffness, and moves material toward the elements that add most
while holding the total volume at the target.

What the optimiser moves is a *design parametrisation*: free voxels behind a density filter (`VoxelDesign`, the raw
result), or 2.5D machining layers (`prismatic.Prismatic25D`). Each maps its variables to physical densities and
back-propagates sensitivities; the optimiser itself only sees variables in [0, 1].
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np
from scipy import ndimage

from .fe import AMG, Grid, Springs, Stiffness, solve

E_MIN = 1e-6  # void stiffness, relative to solid
PENAL = 3.0
MOVE = 0.2
# CG tolerance per state solve, relative to the load. 1e-3 gave the same clevis design as 1e-5 (density difference
# < 1e-4, identical solid/void stiffness) in 30% less time; the W-cycle takes fewer CG iterations than V.
SOLVE_RTOL = 1e-3
SOLVE_CYCLE = "W"


@dataclass
class Problem:
    grid: Grid
    void: np.ndarray  # (nx, ny, nz) bool, held at 0 (keep-outs, mating parts, bolt and pin paths)
    solid: np.ndarray  # (nx, ny, nz) bool, held at 1 (material at the interfaces)
    fixed_dofs: np.ndarray
    loads: list[np.ndarray]  # full-length force vectors, one per load case
    filter_radius: float = 1.5  # in elements
    springs: Springs | None = None  # bearing springs at pin and link supports

    @property
    def design(self) -> np.ndarray:
        return ~(self.void | self.solid)


class Design(Protocol):
    free: np.ndarray  # bool mask over the variables: which ones the optimiser may move

    def initial(self, volume: float) -> np.ndarray: ...
    def from_physical(self, x: np.ndarray) -> np.ndarray: ...
    def forward(self, z: np.ndarray) -> tuple[np.ndarray, Any]: ...
    def backward(self, g: np.ndarray, cache: Any) -> np.ndarray: ...


def cone_kernel(r: float, dims: int) -> np.ndarray:
    n = int(np.ceil(r)) - 1
    ax = np.arange(-n, n + 1)
    grids = np.meshgrid(*([ax] * dims), indexing="ij")
    return np.maximum(0.0, r - np.sqrt(sum(g**2 for g in grids)))


class VoxelDesign:
    """Free voxel densities behind a density filter (the classic, unconstrained 'raw' result)."""

    def __init__(self, p: Problem):
        self.p = p
        self.active = ~p.void
        self.free = p.design
        self.kernel = cone_kernel(p.filter_radius, 3)
        self.weight = self._conv(self.active.astype(float))  # filter only spreads among non-void elements
        self.weight[~self.active] = 1.0

    def _conv(self, a):
        return ndimage.convolve(a, self.kernel, mode="constant", cval=0.0)

    def initial(self, volume: float) -> np.ndarray:
        x = np.where(self.free, volume, 0.0)
        x[self.p.solid] = 1.0
        return x

    def from_physical(self, x: np.ndarray) -> np.ndarray:
        return np.where(self.p.void, 0.0, np.where(self.p.solid, 1.0, x))

    def forward(self, z: np.ndarray):
        xf = self._conv(np.where(self.active, z, 0.0)) / self.weight
        xf[self.p.void] = 0.0
        xf[self.p.solid] = 1.0
        return xf, None

    def backward(self, g: np.ndarray, cache) -> np.ndarray:
        g = np.where(self.free, g, 0.0)
        return self._conv(g / self.weight)  # the kernel is symmetric


@dataclass
class Iteration:
    it: int
    compliance: float  # normalised: 1.0 is each case's stiffness at the first iteration, summed / n
    volume: float  # fraction of the available (non-void) volume
    change: float
    seconds: float


@dataclass
class Result:
    x: np.ndarray  # physical densities, (nx, ny, nz)
    raw: np.ndarray  # design variables, for warm-starting the next run with the same parametrisation
    history: list[Iteration] = field(default_factory=list)
    compliance: list[float] = field(default_factory=list)  # absolute F·u per load case, N·mm
    converged: bool = False


class Optimizer:
    """Holds the per-problem set-up (stiffness pattern, preconditioner) so repeated runs reuse it."""

    def __init__(self, problem: Problem, design: Design | None = None):
        self.p = problem
        self.d = design or VoxelDesign(problem)
        self.active = ~problem.void
        self.stiff = Stiffness(problem.grid, self.active, problem.fixed_dofs, springs=problem.springs)
        self.rhs = [self.stiff.reduce(f) for f in problem.loads]
        self.amg = AMG(self.stiff.B)
        self.c0: list[float] | None = None
        self.u: list[np.ndarray] | None = None
        self.available = int(self.active.sum())

    def physical(self, z: np.ndarray) -> np.ndarray:
        return self.d.forward(z)[0]

    def solid_compliance(self, solid: np.ndarray) -> list[float]:
        """F·u per load case for a 0/1 part (elements at or above 0.5 solid), solved tightly: the like-for-like
        stiffness of a raw result cut at 0.5 and a sharp machined part."""
        E = np.where(solid.ravel()[self.stiff.elements] >= 0.5, 1.0, E_MIN)
        u = solve(self.stiff.assemble(E), self.stiff.B, self.rhs, rtol=1e-6, cycle="W")
        return [float(f @ ui) for f, ui in zip(self.rhs, u)]

    def run(self, volume: float, x0: np.ndarray | None = None, max_iter: int = 80, tol: float = 0.01,
            on_iter: Callable[[Iteration, np.ndarray], bool] | None = None) -> Result:
        """Optimise to `volume` (fraction of the non-void volume) from design variables `x0` (or a uniform start).
        `on_iter` returns False to stop early."""
        free = self.d.free
        z = self.d.initial(volume) if x0 is None else x0.copy()
        target = volume * self.available
        res = Result(x=self.physical(z), raw=z)
        ke = self.stiff.KE
        edof = self.stiff.edof
        for it in range(1, max_iter + 1):
            t = time.time()
            xp, cache = self.d.forward(z)
            xa = xp.ravel()[self.stiff.elements]
            E = E_MIN + xa**PENAL * (1 - E_MIN)
            K = self.stiff.assemble(E)
            self.u = solve(K, self.stiff.B, self.rhs, self.u, rtol=SOLVE_RTOL, cycle=SOLVE_CYCLE, amg=self.amg)
            if self.c0 is None:
                self.c0 = [max(float(f @ u), 1e-30) for f, u in zip(self.rhs, self.u)]
            dc = np.zeros(xa.shape)
            comp = []
            for u, f, c0 in zip(self.u, self.rhs, self.c0):
                ue = self.stiff.expand(u)[edof]
                ce = np.einsum("ij,jk,ik->i", ue, ke, ue)
                comp.append(float(f @ u))
                dc += ce / c0
            n_lc = len(self.rhs)
            c_norm = sum(c / c0 for c, c0 in zip(comp, self.c0)) / n_lc
            dc = -PENAL * xa ** (PENAL - 1) * (1 - E_MIN) * dc / n_lc
            dc_full = np.zeros(self.p.grid.shape)
            dc_full.ravel()[self.stiff.elements] = dc
            dc_z = self.d.backward(dc_full, cache)
            dv_z = self.d.backward(self.active.astype(float), cache)
            v_now = float(xp[self.active].sum())
            z_new = self._oc(z, dc_z, dv_z, v_now, free, target, getattr(self.d, "move", MOVE), getattr(self.d, "exact_volume", False))
            change = float(np.abs(z_new - z)[free].max()) if free.any() else 0.0
            z = z_new
            xp = self.physical(z)
            step = Iteration(it, c_norm, float(xp[self.active].sum() / self.available), change, time.time() - t)
            res.history.append(step)
            res.x, res.raw, res.compliance = xp, z, comp
            if on_iter is not None and on_iter(step, xp) is False:
                break
            # Converged when densities stop moving, or (single elements tend to flicker) when the volume is on
            # target and the stiffness has stopped improving.
            hist = res.history
            on_target = abs(step.volume - volume) < 0.005
            flat = len(hist) > 10 and abs(hist[-6].compliance - c_norm) < 1e-3 * c_norm
            if it > 5 and (change < tol or (on_target and flat)):
                res.converged = True
                break
        return res

    def _oc(self, z, dc, dv, v_now, free, target, move=MOVE, exact=False):
        """Optimality-criteria update with bisection on the multiplier.

        The step uses the plain OC rule (dv = 1), which gave ~2% stiffer cantilevers than weighting by the filtered
        volume sensitivity. The volume it lands on is predicted with the true sensitivity, linearised about the
        current design: exact for the density filter. Strongly nonlinear parametrisations (the 2.5D product of
        faces, which undershot the target by 20% this way) set `exact_volume` and are evaluated directly.
        """
        dc = np.minimum(dc, -1e-30)
        lo, hi = 1e-12, 1e12
        z_new = z
        while hi / lo > 1 + 1e-4:
            mid = np.sqrt(lo * hi)
            cand = z * np.sqrt(-dc / mid)
            cand = np.clip(cand, np.maximum(0.0, z - move), np.minimum(1.0, z + move))
            z_new = np.where(free, cand, z)
            v_new = float(self.physical(z_new)[self.active].sum()) if exact else v_now + float((dv * (z_new - z))[free].sum())
            if v_new > target:
                lo = mid
            else:
                hi = mid
        return z_new
