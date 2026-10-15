"""2.5D machining parametrisation: profiles and flat-floored pockets cut from a few setup faces.

Each setup face has a tool direction s (an axis, pointing from the part towards the tool) and K depth layers: equal
slabs of the grid stacked inward from that face. A layer is a 2-D map (over the two axes square to s), extruded
straight through the layer, so walls are parallel to the tool and floors flat at the layer boundaries. A pocket can
only deepen where the layers above it are cut away, so a layer keeps material wherever it or any shallower layer
does: M_k = 1 − Π_{j≤k} (1 − y_j), applied between layers (at most a few), not element by element. (Element by
element, one pale map value ORed down an 18-element slab came out 98% solid, and the optimiser exploited it.)
Removing every layer is a profile cut straight through.

The part is what survives every face: x = Π_faces m_face (exact intersection for 0/1 densities). Each map gets a
2-D cone filter about the cutter radius, which sets the smallest pocket width and the corner radius.

Forced-solid interface material and forced-empty regions (keep-outs, mating parts, drilled bolt and pin paths)
are applied last, as separate operations; `report()` counts the keep-out elements the faces leave uncut.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage

from .simp import Problem, cone_kernel

AXES = "xyz"
MIN_Z = 1e-3  # variables at exactly 0 could never grow back under the multiplicative OC update


@dataclass
class Face:
    axis: int  # 0/1/2
    sign: int  # +1: tool on the high side of the axis (cuts start at the last slice)
    layers: np.ndarray  # (n_axis,) layer index of each slice, 0 = nearest the tool

    @property
    def step(self) -> tuple[int, int, int]:
        s = [0, 0, 0]
        s[self.axis] = self.sign
        return tuple(s)

    @property
    def name(self) -> str:
        return ("+" if self.sign > 0 else "−") + AXES[self.axis].upper()


def setup_faces(rotary_axis: int, open_end: int = 0) -> list[tuple[int, int]]:
    """(axis, sign) of the setup faces for a 4-axis mill indexed every 90° about `rotary_axis`, plus the open end."""
    faces = [(a, s) for a in range(3) if a != rotary_axis for s in (1, -1)]
    if open_end:
        faces.append((rotary_axis, 1 if open_end > 0 else -1))
    return faces


class Prismatic25D:
    exact_volume = True  # the product over faces is too nonlinear for a linearised volume (cheap to evaluate: ~1 ms)
    move = 0.1

    def __init__(self, p: Problem, faces: list[tuple[int, int]], depths: int, filter_radius: float):
        self.p = p
        self.shape = p.grid.shape
        self.depths = depths
        self.faces: list[Face] = []
        self.map_shapes: list[tuple[int, int, int]] = []
        for axis, sign in faces:
            n = self.shape[axis]
            depth_from_tool = (n - 1 - np.arange(n)) if sign > 0 else np.arange(n)
            layers = np.minimum(depth_from_tool * depths // n, depths - 1)
            self.faces.append(Face(axis, sign, layers))
            other = [self.shape[i] for i in range(3) if i != axis]
            self.map_shapes.append((depths, other[0], other[1]))
        self.sizes = [int(np.prod(s)) for s in self.map_shapes]
        self.offsets = np.cumsum([0] + self.sizes)
        self.free = np.ones(int(self.offsets[-1]), bool)
        self.kernel = cone_kernel(filter_radius, 2)[None]  # 2-D, applied to each layer separately
        self.weights = [ndimage.convolve(np.ones(s), self.kernel, mode="constant") for s in self.map_shapes]

    # ---------- variables <-> maps ----------

    def maps(self, z: np.ndarray) -> list[np.ndarray]:
        return [z[self.offsets[i]:self.offsets[i + 1]].reshape(s) for i, s in enumerate(self.map_shapes)]

    def flat(self, maps: list[np.ndarray]) -> np.ndarray:
        return np.concatenate([m.ravel() for m in maps])

    def _extrude(self, f: Face, m: np.ndarray) -> np.ndarray:
        """Layer maps (K, A, B) to a 3-D field: each slice along the face axis takes its layer's map."""
        return np.moveaxis(m[f.layers], 0, f.axis)

    def _extrude_adjoint(self, f: Face, g: np.ndarray) -> np.ndarray:
        out = np.zeros((self.depths,) + tuple(s for i, s in enumerate(self.shape) if i != f.axis))
        np.add.at(out, f.layers, np.moveaxis(g, f.axis, 0))
        return out

    # ---------- the parametrisation ----------

    def initial(self, volume: float) -> np.ndarray:
        # Π over faces of a uniform y gives y^n_faces: start every map at the n-th root of the target.
        return np.full(self.free.shape, max(MIN_Z, volume ** (1 / len(self.faces))))

    def from_physical(self, x: np.ndarray) -> np.ndarray:
        """Warm start from a 3-D field: each layer map keeps the most material anywhere along its slab, so the part
        starts as the smallest 2.5D shape containing `x` (more than the target volume; the optimiser trims it)."""
        x = np.where(self.p.void, 0.0, x)
        maps = []
        for f, shape in zip(self.faces, self.map_shapes):
            mv = np.moveaxis(x, f.axis, 0)
            m = np.zeros(shape)
            np.maximum.at(m, f.layers, mv)
            maps.append(m)
        return np.maximum(self.flat(maps), MIN_Z)

    def _filtered(self, i: int, m: np.ndarray) -> np.ndarray:
        return ndimage.convolve(m, self.kernel, mode="constant") / self.weights[i]

    @staticmethod
    def _cover(y: np.ndarray) -> np.ndarray:
        """Layer k keeps material if it or any shallower layer does: 1 − Π_{j≤k}(1 − y_j)."""
        return 1.0 - np.cumprod(1.0 - y, axis=0)

    @staticmethod
    def _cover_adjoint(y: np.ndarray, g: np.ndarray) -> np.ndarray:
        """dL/dy_j = Σ_{k≥j} g_k · Π_{i≤k, i≠j} (1 − y_i), without dividing by (1 − y_j)."""
        K = y.shape[0]
        one = 1.0 - y
        out = np.zeros_like(y)
        for j in range(K):
            before = np.prod(one[:j], axis=0) if j else np.ones_like(y[0])
            run = before.copy()  # Π over i ≤ k, i ≠ j, built up as k grows
            for k in range(j, K):
                if k > j:
                    run = run * one[k]
                out[j] += g[k] * run
        return out

    def forward(self, z: np.ndarray):
        ys, ms = [], []
        for i, (f, m) in enumerate(zip(self.faces, self.maps(z))):
            y = self._filtered(i, m)
            ys.append(y)
            ms.append(self._extrude(f, self._cover(y)))
        x = np.prod(ms, axis=0)
        x[self.p.void] = 0.0
        x[self.p.solid] = 1.0
        return x, (ys, ms)

    def backward(self, g: np.ndarray, cache) -> np.ndarray:
        ys, ms = cache
        g = np.where(self.p.design, g, 0.0)
        out = []
        for i, f in enumerate(self.faces):
            others = np.ones_like(g)
            for j, m in enumerate(ms):
                if j != i:
                    others *= m
            gM = self._extrude_adjoint(f, g * others)
            gy = self._cover_adjoint(ys[i], gM)
            out.append(ndimage.convolve(gy / self.weights[i], self.kernel, mode="constant"))
        return self.flat(out)

    def sharp(self, z: np.ndarray, cut: float = 0.5) -> np.ndarray:
        """The machined part as 0/1: each filtered layer map cut at `cut`, extruded and intersected with no blur, so
        walls are exactly parallel to the cutter and floors flat."""
        ms = [self._extrude(f, self._cover((self._filtered(i, m) >= cut).astype(float)))
              for i, (f, m) in enumerate(zip(self.faces, self.maps(z)))]
        x = np.prod(ms, axis=0)
        x[self.p.void] = 0.0
        x[self.p.solid] = 1.0
        return x

    # ---------- reporting ----------

    def describe(self) -> str:
        return f"2.5D from {', '.join(f.name for f in self.faces)}, {self.depths} floor depths each"
