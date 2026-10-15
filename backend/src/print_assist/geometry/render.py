"""A small software renderer: a part (one or more bodies) to a shaded PNG with its edges drawn, from a standard view.

It needs no GPU, browser or display, so the AI (through the MCP server or the in-app chat) can look at the model the
same way on a laptop or a headless home server. Orthographic projection, z-buffered triangles with two-light Lambert
shading, B-rep edges drawn where they are not hidden.
"""

from __future__ import annotations

import struct
import zlib

import numpy as np
from build123d import Shape

VIEWS = {  # direction the camera looks along (from the eye toward the part), and screen-up
    "iso": ((-1.0, 1.0, -0.8), (0.0, 0.0, 1.0)),
    "front": ((0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
    "back": ((0.0, -1.0, 0.0), (0.0, 0.0, 1.0)),
    "right": ((-1.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
    "left": ((1.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
    "top": ((0.0, 0.0, -1.0), (0.0, 1.0, 0.0)),
    "bottom": ((0.0, 0.0, 1.0), (0.0, -1.0, 0.0)),
}
BACKGROUND = np.array([42, 46, 53], dtype=np.float32)
DEFAULT_COLOR = (0.80, 0.78, 0.72)
EDGE = np.array([24, 26, 30], dtype=np.float32)


def _hex(color: str | None) -> tuple[float, float, float]:
    if not color or not color.startswith("#") or len(color) != 7:
        return DEFAULT_COLOR
    return tuple(int(color[i:i + 2], 16) / 255 for i in (1, 3, 5))  # type: ignore[return-value]


def _basis(view: str):
    d, up = VIEWS[view]
    f = np.array(d, dtype=np.float64)
    f /= np.linalg.norm(f)
    u = np.array(up, dtype=np.float64)
    r = np.cross(f, u)
    r /= np.linalg.norm(r)
    u = np.cross(r, f)
    return r, u, f


def render(bodies: list[tuple[Shape, str | None]], view: str = "iso", size: int = 640, tolerance: float = 0.05,
           meshes: list[tuple[np.ndarray, np.ndarray, str | None]] = (),
           region: tuple[float, float, float, float, float, float] | None = None) -> bytes:
    """PNG bytes of the bodies (shape, "#rrggbb" colour) and triangle meshes (points, triangles, colour) seen from
    `view`. `region` (xmin, ymin, zmin, xmax, ymax, zmax) keeps only what lies inside it and frames it."""
    if view not in VIEWS:
        raise ValueError(f"unknown view {view!r}; views: {', '.join(VIEWS)}")
    size = max(64, min(int(size), 2048))
    r, u, f = _basis(view)
    tris, cols, norms, lines = [], [], [], []
    for shape, color in bodies:
        _shape_triangles(shape, color, tolerance, tris, cols, norms, lines)
    for points, faces, color in meshes:
        _mesh_triangles(points, faces, color, tris, cols, norms, lines)
    if region is not None and tris:
        lo, hi = np.array(region[:3]), np.array(region[3:])
        keep = [((t.mean(1) >= lo) & (t.mean(1) <= hi)).all(1) for t in tris]
        tris = [t[k] for t, k in zip(tris, keep)]
        cols = [c[k] for c, k in zip(cols, keep)]
        norms = [n[k] for n, k in zip(norms, keep)]
        lines = [L for L in (_clip(L, lo, hi) for L in lines) if L is not None]
        tris = [t for t in tris if len(t)]
        cols = [c for c in cols if len(c)]
        norms = [n for n in norms if len(n)]
    return _raster(tris, cols, norms, lines, r, u, f, size, tolerance)


def _clip(L: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray | None:
    inside = ((L >= lo) & (L <= hi)).all(1)
    return L[inside] if inside.sum() >= 2 else None


def _shape_triangles(shape: Shape, color, tolerance, tris, cols, norms, lines) -> None:
    c = np.array(_hex(color), dtype=np.float32)
    for face in shape.faces():
        try:
            verts, faces = face.tessellate(tolerance, 0.3)
        except Exception:  # noqa: BLE001 - a sliver face the mesher cannot triangulate
            continue
        if not faces:
            continue
        v = np.array([[p.X, p.Y, p.Z] for p in verts], dtype=np.float64)
        try:  # the surface's own normal at each vertex: smooth shading on curved faces
            vn = np.array([list(face.normal_at(p)) for p in verts], dtype=np.float64)
        except Exception:  # noqa: BLE001
            vn = None
        idx = np.array(faces, dtype=np.int64)
        t = v[idx]
        if vn is None:
            fn = np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0])
            fn /= np.maximum(np.linalg.norm(fn, axis=1, keepdims=True), 1e-12)
            tn = np.repeat(fn[:, None, :], 3, axis=1)
        else:
            tn = vn[idx]
        tris.append(t)
        norms.append(tn)
        cols.append(np.tile(c, (len(t), 1)))
    for e in shape.edges():
        n = max(2, min(64, int(e.length / max(tolerance * 4, 0.2)) + 2))
        pts = [e.position_at(i / (n - 1)) for i in range(n)]
        lines.append(np.array([[p.X, p.Y, p.Z] for p in pts], dtype=np.float64))


def _mesh_triangles(points: np.ndarray, faces: np.ndarray, color, tris, cols, norms, lines, crease: float = 30.0) -> None:
    """A triangle mesh, flat shaded, with its creases (faces meeting at more than `crease` degrees) and open edges
    drawn: the mesh's own triangle edges would hide the shape."""
    if not len(faces):
        return
    t = points[faces]
    fn = np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0])
    fn /= np.maximum(np.linalg.norm(fn, axis=1, keepdims=True), 1e-12)
    tris.append(t)
    norms.append(np.repeat(fn[:, None, :], 3, axis=1))
    cols.append(np.tile(np.array(_hex(color), dtype=np.float32), (len(t), 1)))
    e = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    owner = np.tile(np.arange(len(faces)), 3)
    key = np.sort(e, axis=1)
    order = np.lexsort((key[:, 1], key[:, 0]))
    key, owner = key[order], owner[order]
    same = (key[1:] == key[:-1]).all(1)
    i = np.nonzero(same)[0]  # pairs (i, i+1) share an edge
    cos = (fn[owner[i]] * fn[owner[i + 1]]).sum(1)
    sharp = key[i][cos < np.cos(np.radians(crease))]
    paired = np.zeros(len(key), bool)
    paired[i] = paired[i + 1] = True
    open_ = key[~paired]
    for a, b in np.concatenate([sharp, open_]) if len(sharp) + len(open_) else ():
        lines.append(points[[a, b]])


def _raster(tris, cols, norms, lines, r, u, f, size, tolerance) -> bytes:
    img = np.empty((size, size, 3), dtype=np.float32)
    img[:] = BACKGROUND
    if not tris:
        return _png(img)
    T = np.concatenate(tris)
    C = np.concatenate(cols)

    # View space: x right, y up, z depth (larger = farther).
    allv = T.reshape(-1, 3)
    centre = (allv.min(0) + allv.max(0)) / 2
    def to_view(p):
        q = p - centre
        return np.stack([q @ r, q @ u, q @ f], axis=-1)
    V = to_view(T)
    span = max(np.ptp(V[..., 0]), np.ptp(V[..., 1]), 1e-6) * 1.12
    scale = size / span
    sx = V[..., 0] * scale + size / 2
    sy = size / 2 - V[..., 1] * scale
    sz = V[..., 2]

    # Shading per vertex from the surface normals (two lights in view space), lit on both sides.
    N = np.concatenate(norms)  # (n, 3, 3)
    nv = np.stack([N @ r, N @ u, N @ f], axis=-1)
    nv = np.where(nv[..., 2:3] > 0, -nv, nv)
    key = np.array([-0.45, 0.6, -0.66])
    fill = np.array([0.6, -0.2, -0.77])
    key /= np.linalg.norm(key)
    fill /= np.linalg.norm(fill)
    shade = 0.28 + 0.62 * np.clip(nv @ key, 0, 1) + 0.22 * np.clip(nv @ fill, 0, 1)  # (n, 3)

    zbuf = np.full((size, size), np.inf, dtype=np.float64)
    xmin = np.clip(np.floor(sx.min(1)), 0, size - 1).astype(int)
    xmax = np.clip(np.ceil(sx.max(1)), 0, size - 1).astype(int)
    ymin = np.clip(np.floor(sy.min(1)), 0, size - 1).astype(int)
    ymax = np.clip(np.ceil(sy.max(1)), 0, size - 1).astype(int)
    # Small triangles (dense meshes) are rasterised in batches over a fixed window of pixels; the loop below takes
    # the rest one by one.
    W = 5
    small = ((xmax - xmin) < W) & ((ymax - ymin) < W)
    idx = np.nonzero(small)[0]
    off = np.array([(dx, dy) for dy in range(W) for dx in range(W)])
    for c0 in range(0, len(idx), 40000):
        k = idx[c0:c0 + 40000]
        px = xmin[k][:, None] + off[None, :, 0]  # (n, W*W)
        py = ymin[k][:, None] + off[None, :, 1]
        cx, cy = px + 0.5, py + 0.5
        ax, ay, bx, by, qx, qy = (sx[k, 0:1], sy[k, 0:1], sx[k, 1:2], sy[k, 1:2], sx[k, 2:3], sy[k, 2:3])
        den = (by - qy) * (ax - qx) + (qx - bx) * (ay - qy)
        den = np.where(np.abs(den) < 1e-12, np.nan, den)
        w0 = ((by - qy) * (cx - qx) + (qx - bx) * (cy - qy)) / den
        w1 = ((qy - ay) * (cx - qx) + (ax - qx) * (cy - qy)) / den
        w2 = 1 - w0 - w1
        inside = (w0 >= -1e-6) & (w1 >= -1e-6) & (w2 >= -1e-6) & (px < size) & (py < size)
        # a sub-pixel triangle covering no pixel centre still shows: take its nearest pixel
        none = ~inside.any(1)
        if none.any():
            inside[none, 0] = True
            w0[none, 0] = w1[none, 0] = w2[none, 0] = 1 / 3
        t_i, c_i = np.nonzero(inside)
        fx, fy = px[t_i, c_i], py[t_i, c_i]
        W0, W1, W2 = w0[t_i, c_i], w1[t_i, c_i], w2[t_i, c_i]
        tri = k[t_i]
        fz = W0 * sz[tri, 0] + W1 * sz[tri, 1] + W2 * sz[tri, 2]
        flat = fy * size + fx
        order = np.lexsort((fz, flat))
        first = np.ones(len(order), bool)
        first[1:] = flat[order][1:] != flat[order][:-1]
        o = order[first]
        fx, fy, fz, tri = fx[o], fy[o], fz[o], tri[o]
        sh = W0[o] * shade[tri, 0] + W1[o] * shade[tri, 1] + W2[o] * shade[tri, 2]
        nearer = fz < zbuf[fy, fx]
        fx, fy, fz, tri, sh = fx[nearer], fy[nearer], fz[nearer], tri[nearer], sh[nearer]
        zbuf[fy, fx] = fz
        img[fy, fx] = np.clip(C[tri] * sh[:, None], 0, 1) * 255
    tiny = small
    for i in np.nonzero(~tiny)[0]:
        x0, x1, y0, y1 = xmin[i], xmax[i], ymin[i], ymax[i]
        if x1 < x0 or y1 < y0:
            continue
        px, py = np.meshgrid(np.arange(x0, x1 + 1) + 0.5, np.arange(y0, y1 + 1) + 0.5)
        ax, ay, bx, by, cx, cy = sx[i, 0], sy[i, 0], sx[i, 1], sy[i, 1], sx[i, 2], sy[i, 2]
        den = (by - cy) * (ax - cx) + (cx - bx) * (ay - cy)
        if abs(den) < 1e-12:
            continue
        w0 = ((by - cy) * (px - cx) + (cx - bx) * (py - cy)) / den
        w1 = ((cy - ay) * (px - cx) + (ax - cx) * (py - cy)) / den
        w2 = 1 - w0 - w1
        inside = (w0 >= -1e-6) & (w1 >= -1e-6) & (w2 >= -1e-6)
        if not inside.any():
            continue
        z = w0 * sz[i, 0] + w1 * sz[i, 1] + w2 * sz[i, 2]
        win = zbuf[y0:y1 + 1, x0:x1 + 1]
        hit = inside & (z < win)
        if not hit.any():
            continue
        win[hit] = z[hit]
        sh = (w0 * shade[i, 0] + w1 * shade[i, 1] + w2 * shade[i, 2])[hit]
        img[y0:y1 + 1, x0:x1 + 1][hit] = np.clip(C[i][None, :] * sh[:, None], 0, 1) * 255

    # Edges: every segment sampled at pixel spacing, all at once; drawn where not behind the surface.
    if lines:
        eps = span * 0.004
        A = np.concatenate([L[:-1] for L in lines])
        B = np.concatenate([L[1:] for L in lines])
        a, b = to_view(A), to_view(B)
        ax_, ay_ = a[:, 0] * scale + size / 2, size / 2 - a[:, 1] * scale
        bx_, by_ = b[:, 0] * scale + size / 2, size / 2 - b[:, 1] * scale
        k = (np.maximum(np.abs(bx_ - ax_), np.abs(by_ - ay_)).astype(int) + 2).clip(2, 4 * size)
        seg = np.repeat(np.arange(len(k)), k)
        t = (np.arange(len(seg)) - np.repeat(np.cumsum(k) - k, k)) / np.repeat(k - 1, k)
        xs = (ax_[seg] + (bx_ - ax_)[seg] * t).astype(int)
        ys = (ay_[seg] + (by_ - ay_)[seg] * t).astype(int)
        zs = a[seg, 2] + (b[:, 2] - a[:, 2])[seg] * t
        ok = (xs >= 0) & (xs < size) & (ys >= 0) & (ys < size)
        xs, ys, zs = xs[ok], ys[ok], zs[ok]
        vis = zs <= zbuf[ys, xs] + eps
        img[ys[vis], xs[vis]] = EDGE
    return _png(img)


def _png(img: np.ndarray) -> bytes:
    h, w, _ = img.shape
    raw = np.clip(img, 0, 255).astype(np.uint8)
    rows = b"".join(b"\x00" + raw[y].tobytes() for y in range(h))

    def chunk(tag: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(rows, 6)) + chunk(b"IEND", b""))


def png_size(data: bytes) -> tuple[int, int]:
    return struct.unpack(">II", data[16:24])


