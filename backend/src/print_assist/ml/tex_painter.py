"""Local 3D Texture Painter for Character Mode (`Hunyuan3D-Paint` on Apple Silicon MPS, CUDA, and CPU).

Renders exact 3D surface normal and world-position maps of the sculpted mesh using pure-NumPy/PyTorch
cross-platform rasterizer and vertex-inpaint shims (ZERO C++/CUDA compiler required on macOS or Windows!),
runs `hunyuan3d-paint-v2-0-turbo` multi-view diffusion conditioned on both the approved concept image AND
the mesh's own 3D geometry, and bakes a seamless UV texture map directly onto the `.glb` mesh.

Invoked inside `ml_root()/venv` as an isolated subprocess so GPU/unified memory is 100% released on exit.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Callable

import numpy as np
from PIL import Image

from . import gpu_job, ml_job_update, ml_root
from .hardware import detect_hardware
from .installer import installer, sculptor_python, tex_painter_dir, tex_painter_ready


class TexPainterError(RuntimeError):
    pass


def rasterize_clip_space_numpy(
    v_clip: np.ndarray,
    faces: np.ndarray,
    height: int,
    width: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Pure-NumPy drop-in equivalent of `custom_rasterizer_kernel.rasterize_image` (works on macOS, Windows, Linux).

    Matches `rasterizer_gpu.cu` coordinate conventions pixel-for-pixel:
      - `vx = (V_x / V_w * 0.5 + 0.5) * (W - 1) + 0.5`
      - `vy = (0.5 + 0.5 * V_y / V_w) * (H - 1) + 0.5`
      - `vz = V_z / V_w * 0.49999 + 0.5` (smaller z wins)
      - Pixel center at `(px + 0.5, py + 0.5)`
      - Returns `(findices [H, W] int32 1-indexed (0=bg), barycentric [H, W, 3] float32 perspective-correct)`.
    """
    V = np.asarray(v_clip, dtype=np.float32)
    F = np.asarray(faces, dtype=np.int32)
    if V.ndim == 3:
        V = V[0]

    findices = np.zeros((height, width), dtype=np.int32)
    bary = np.zeros((height, width, 3), dtype=np.float32)
    if len(V) == 0 or len(F) == 0 or height <= 0 or width <= 0:
        return findices, bary

    w_safe = np.where(np.abs(V[:, 3]) < 1e-12, 1e-12, V[:, 3]).astype(np.float32)
    inv_w = 1.0 / w_safe
    vx = (V[:, 0] * inv_w * 0.5 + 0.5) * (width - 1) + 0.5
    vy = (0.5 + 0.5 * V[:, 1] * inv_w) * (height - 1) + 0.5
    vz = V[:, 2] * inv_w * 0.49999 + 0.5

    sx = vx[F]  # (N_f, 3)
    sy = vy[F]
    sz = vz[F]

    xmin = np.floor(sx.min(axis=1)).astype(np.int32)
    xmax = np.ceil(sx.max(axis=1)).astype(np.int32)
    ymin = np.floor(sy.min(axis=1)).astype(np.int32)
    ymax = np.ceil(sy.max(axis=1)).astype(np.int32)

    valid_tri = (xmax >= 0) & (xmin < width) & (ymax >= 0) & (ymin < height)
    xmin = np.clip(xmin, 0, width - 1)
    xmax = np.clip(xmax, 0, width - 1)
    ymin = np.clip(ymin, 0, height - 1)
    ymax = np.clip(ymax, 0, height - 1)

    zbuf = np.full((height, width), np.inf, dtype=np.float32)

    # Vectorized fast path for small triangles (covers 95%+ of organic sculpts)
    W_box = 8
    small = valid_tri & ((xmax - xmin) < W_box) & ((ymax - ymin) < W_box)
    s_idx = np.nonzero(small)[0]
    if len(s_idx) > 0:
        off = np.array([(dx, dy) for dy in range(W_box) for dx in range(W_box)], dtype=np.int32)
        for c0 in range(0, len(s_idx), 30000):
            k = s_idx[c0:c0 + 30000]
            px = xmin[k][:, None] + off[None, :, 0]
            py = ymin[k][:, None] + off[None, :, 1]
            cx = px.astype(np.float32) + 0.5
            cy = py.astype(np.float32) + 0.5

            ax, ay = sx[k, 0:1], sy[k, 0:1]
            bx, by = sx[k, 1:2], sy[k, 1:2]
            qx, qy = sx[k, 2:3], sy[k, 2:3]
            den = (by - qy) * (ax - qx) + (qx - bx) * (ay - qy)
            den = np.where(np.abs(den) < 1e-10, np.nan, den)
            w0 = ((by - qy) * (cx - qx) + (qx - bx) * (cy - qy)) / den
            w1 = ((qy - ay) * (cx - qx) + (ax - qx) * (cy - qy)) / den
            w2 = 1.0 - w0 - w1

            inside = (w0 >= -1e-5) & (w1 >= -1e-5) & (w2 >= -1e-5) & (px < width) & (py < height)
            t_i, c_i = np.nonzero(inside)
            if len(t_i) == 0:
                continue
            fx = px[t_i, c_i]
            fy = py[t_i, c_i]
            tri = k[t_i]
            fz = (w0[t_i, c_i] * sz[tri, 0] + w1[t_i, c_i] * sz[tri, 1] + w2[t_i, c_i] * sz[tri, 2]).astype(np.float32)
            flat = fy.astype(np.int64) * width + fx.astype(np.int64)
            order = np.lexsort((fz, flat))
            first = np.ones(len(order), dtype=bool)
            first[1:] = flat[order][1:] != flat[order][:-1]
            o = order[first]
            fx, fy, fz, tri = fx[o], fy[o], fz[o], tri[o]
            nearer = fz < zbuf[fy, fx]
            if not np.any(nearer):
                continue
            fx, fy, fz, tri = fx[nearer], fy[nearer], fz[nearer], tri[nearer]
            zbuf[fy, fx] = fz
            findices[fy, fx] = tri.astype(np.int32) + 1

    # Tight per-triangle bbox loop for larger triangles
    for i in np.nonzero(valid_tri & ~small)[0]:
        x0_i, x1_i, y0_i, y1_i = int(xmin[i]), int(xmax[i]), int(ymin[i]), int(ymax[i])
        if x1_i < x0_i or y1_i < y0_i:
            continue
        ax, ay, bx, by, qx, qy = float(sx[i, 0]), float(sy[i, 0]), float(sx[i, 1]), float(sy[i, 1]), float(sx[i, 2]), float(sy[i, 2])
        den = (by - qy) * (ax - qx) + (qx - bx) * (ay - qy)
        if abs(den) < 1e-10:
            continue
        px, py = np.meshgrid(
            np.arange(x0_i, x1_i + 1, dtype=np.float32) + 0.5,
            np.arange(y0_i, y1_i + 1, dtype=np.float32) + 0.5,
        )
        w0 = ((by - qy) * (px - qx) + (qx - bx) * (py - qy)) / den
        w1 = ((qy - ay) * (px - qx) + (ax - qx) * (py - qy)) / den
        w2 = 1.0 - w0 - w1
        inside = (w0 >= -1e-5) & (w1 >= -1e-5) & (w2 >= -1e-5)
        if not np.any(inside):
            continue
        z = (w0 * sz[i, 0] + w1 * sz[i, 1] + w2 * sz[i, 2]).astype(np.float32)
        win = zbuf[y0_i:y1_i + 1, x0_i:x1_i + 1]
        hit = inside & (z < win)
        if not np.any(hit):
            continue
        win[hit] = z[hit]
        findices[y0_i:y1_i + 1, x0_i:x1_i + 1][hit] = int(i) + 1

    # Vectorized perspective-correct barycentric coordinates for all visible pixels
    hy, hx = np.nonzero(findices > 0)
    if len(hy) > 0:
        f = findices[hy, hx] - 1
        cx = hx.astype(np.float32) + 0.5
        cy = hy.astype(np.float32) + 0.5
        ax, ay = sx[f, 0], sy[f, 0]
        bx, by = sx[f, 1], sy[f, 1]
        qx, qy = sx[f, 2], sy[f, 2]
        den = (by - qy) * (ax - qx) + (qx - bx) * (ay - qy)
        den = np.where(np.abs(den) < 1e-10, 1e-10, den)
        b0 = ((by - qy) * (cx - qx) + (qx - bx) * (cy - qy)) / den
        b1 = ((qy - ay) * (cx - qx) + (ax - qx) * (cy - qy)) / den
        b2 = 1.0 - b0 - b1

        # Perspective correction matching rasterizer_gpu.cu lines 69-75
        b0 = b0 * inv_w[F[f, 0]]
        b1 = b1 * inv_w[F[f, 1]]
        b2 = b2 * inv_w[F[f, 2]]
        bsum = b0 + b1 + b2
        bsum = np.where(np.abs(bsum) < 1e-12, 1.0, bsum)
        bary[hy, hx, 0] = b0 / bsum
        bary[hy, hx, 1] = b1 / bsum
        bary[hy, hx, 2] = b2 / bsum

    return findices, bary


def mesh_vertice_inpaint_numpy(
    texture: np.ndarray,
    mask: np.ndarray,
    vtx_pos: np.ndarray,
    vtx_uv: np.ndarray,
    pos_idx: np.ndarray,
    uv_idx: np.ndarray,
    method: str = "smooth",
) -> tuple[np.ndarray, np.ndarray]:
    """Pure-NumPy vectorized drop-in equivalent of `mesh_processor.meshVerticeInpaint` (`mesh_processor.cpp`)."""
    tex = np.asarray(texture, dtype=np.float32)
    msk = np.asarray(mask, dtype=np.uint8)
    vpos = np.asarray(vtx_pos, dtype=np.float32)
    vuv = np.asarray(vtx_uv, dtype=np.float32)
    pidx = np.asarray(pos_idx, dtype=np.int32)
    uidx = np.asarray(uv_idx, dtype=np.int32)

    H, W, C = tex.shape
    n_v = len(vpos)
    new_tex = tex.copy()
    new_msk = msk.copy()
    if n_v == 0 or len(pidx) == 0:
        return new_tex, new_msk

    uv_flat = uidx.reshape(-1)
    pos_flat = pidx.reshape(-1)
    uv_v = np.clip(np.rint(vuv[uv_flat, 0] * (W - 1)).astype(np.int32), 0, W - 1)
    uv_u = np.clip(np.rint((1.0 - vuv[uv_flat, 1]) * (H - 1)).astype(np.int32), 0, H - 1)

    vtx_mask = np.zeros(n_v, dtype=bool)
    vtx_color = np.zeros((n_v, C), dtype=np.float32)
    valid_hit = msk[uv_u, uv_v] > 0
    if np.any(valid_hit):
        vh_pos = pos_flat[valid_hit]
        vtx_mask[vh_pos] = True
        vtx_color[vh_pos] = tex[uv_u[valid_hit], uv_v[valid_hit]]

    # Undirected mesh adjacency with inverse-square-distance weights
    e_src = np.concatenate([pidx[:, 0], pidx[:, 1], pidx[:, 2], pidx[:, 1], pidx[:, 2], pidx[:, 0]])
    e_dst = np.concatenate([pidx[:, 1], pidx[:, 2], pidx[:, 0], pidx[:, 0], pidx[:, 1], pidx[:, 2]])
    dist = np.linalg.norm(vpos[e_src] - vpos[e_dst], axis=1)
    e_w = (1.0 / np.maximum(dist, 1e-4)) ** 2

    # Propagate colors smoothly across uncolored vertices until convergence (cap iterations for safety)
    for _ in range(120):
        need = (~vtx_mask[e_src]) & vtx_mask[e_dst]
        if not np.any(need):
            break
        u = e_src[need]
        v = e_dst[need]
        w = e_w[need]
        w_sum = np.zeros(n_v, dtype=np.float32)
        c_sum = np.zeros((n_v, C), dtype=np.float32)
        np.add.at(w_sum, u, w)
        np.add.at(c_sum, u, vtx_color[v] * w[:, None])
        filled = w_sum > 0.0
        vtx_color[filled] = c_sum[filled] / w_sum[filled, None]
        vtx_mask[filled] = True

    colored_corner = vtx_mask[pos_flat]
    if np.any(colored_corner):
        cu = uv_u[colored_corner]
        cv = uv_v[colored_corner]
        cp = pos_flat[colored_corner]
        new_tex[cu, cv] = vtx_color[cp]
        new_msk[cu, cv] = 255

    return new_tex, new_msk


_WORKER_SCRIPT = r'''
import gc, json, os, re, shutil, sys, types
from pathlib import Path
import numpy as np
from PIL import Image

cfg = json.loads(sys.argv[1])
repo_dir = Path(cfg["repo_dir"])
sys.path.insert(0, str(repo_dir))

def emit(stage, pct=0, detail=""):
    print(json.dumps({"stage": stage, "pct": pct, "detail": detail}), flush=True)

import torch

if cfg["accelerator"] == "cuda" and torch.cuda.is_available():
    device = "cuda"
elif cfg["accelerator"] == "mps" and getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

dtype = torch.float32 if device == "cpu" else torch.float16

# ---------- Pure-NumPy/PyTorch cross-platform shims (zero CUDA/C++ extension required!) ----------
def _rasterize_numpy(v_clip, faces, height, width):
    V = np.asarray(v_clip, dtype=np.float32)
    F = np.asarray(faces, dtype=np.int32)
    if V.ndim == 3:
        V = V[0]
    findices = np.zeros((height, width), dtype=np.int32)
    bary = np.zeros((height, width, 3), dtype=np.float32)
    if len(V) == 0 or len(F) == 0:
        return findices, bary
    w_safe = np.where(np.abs(V[:, 3]) < 1e-12, 1e-12, V[:, 3]).astype(np.float32)
    inv_w = 1.0 / w_safe
    vx = (V[:, 0] * inv_w * 0.5 + 0.5) * (width - 1) + 0.5
    vy = (0.5 + 0.5 * V[:, 1] * inv_w) * (height - 1) + 0.5
    vz = V[:, 2] * inv_w * 0.49999 + 0.5
    sx, sy, sz = vx[F], vy[F], vz[F]
    xmin = np.floor(sx.min(axis=1)).astype(np.int32)
    xmax = np.ceil(sx.max(axis=1)).astype(np.int32)
    ymin = np.floor(sy.min(axis=1)).astype(np.int32)
    ymax = np.ceil(sy.max(axis=1)).astype(np.int32)
    valid_tri = (xmax >= 0) & (xmin < width) & (ymax >= 0) & (ymin < height)
    xmin = np.clip(xmin, 0, width - 1)
    xmax = np.clip(xmax, 0, width - 1)
    ymin = np.clip(ymin, 0, height - 1)
    ymax = np.clip(ymax, 0, height - 1)
    zbuf = np.full((height, width), np.inf, dtype=np.float32)

    W_box = 8
    small = valid_tri & ((xmax - xmin) < W_box) & ((ymax - ymin) < W_box)
    s_idx = np.nonzero(small)[0]
    if len(s_idx) > 0:
        off = np.array([(dx, dy) for dy in range(W_box) for dx in range(W_box)], dtype=np.int32)
        for c0 in range(0, len(s_idx), 30000):
            k = s_idx[c0:c0 + 30000]
            px = xmin[k][:, None] + off[None, :, 0]
            py = ymin[k][:, None] + off[None, :, 1]
            cx = px.astype(np.float32) + 0.5
            cy = py.astype(np.float32) + 0.5
            ax, ay = sx[k, 0:1], sy[k, 0:1]
            bx, by = sx[k, 1:2], sy[k, 1:2]
            qx, qy = sx[k, 2:3], sy[k, 2:3]
            den = (by - qy) * (ax - qx) + (qx - bx) * (ay - qy)
            den = np.where(np.abs(den) < 1e-10, np.nan, den)
            w0 = ((by - qy) * (cx - qx) + (qx - bx) * (cy - qy)) / den
            w1 = ((qy - ay) * (cx - qx) + (ax - qx) * (cy - qy)) / den
            w2 = 1.0 - w0 - w1
            inside = (w0 >= -1e-5) & (w1 >= -1e-5) & (w2 >= -1e-5) & (px < width) & (py < height)
            t_i, c_i = np.nonzero(inside)
            if len(t_i) == 0:
                continue
            fx, fy, tri = px[t_i, c_i], py[t_i, c_i], k[t_i]
            fz = (w0[t_i, c_i] * sz[tri, 0] + w1[t_i, c_i] * sz[tri, 1] + w2[t_i, c_i] * sz[tri, 2]).astype(np.float32)
            flat = fy.astype(np.int64) * width + fx.astype(np.int64)
            order = np.lexsort((fz, flat))
            first = np.ones(len(order), dtype=bool)
            first[1:] = flat[order][1:] != flat[order][:-1]
            o = order[first]
            fx, fy, fz, tri = fx[o], fy[o], fz[o], tri[o]
            nearer = fz < zbuf[fy, fx]
            if not np.any(nearer):
                continue
            fx, fy, fz, tri = fx[nearer], fy[nearer], fz[nearer], tri[nearer]
            zbuf[fy, fx] = fz
            findices[fy, fx] = tri.astype(np.int32) + 1

    for i in np.nonzero(valid_tri & ~small)[0]:
        x0_i, x1_i, y0_i, y1_i = int(xmin[i]), int(xmax[i]), int(ymin[i]), int(ymax[i])
        if x1_i < x0_i or y1_i < y0_i:
            continue
        ax, ay, bx, by, qx, qy = float(sx[i, 0]), float(sy[i, 0]), float(sx[i, 1]), float(sy[i, 1]), float(sx[i, 2]), float(sy[i, 2])
        den = (by - qy) * (ax - qx) + (qx - bx) * (ay - qy)
        if abs(den) < 1e-10:
            continue
        px, py = np.meshgrid(
            np.arange(x0_i, x1_i + 1, dtype=np.float32) + 0.5,
            np.arange(y0_i, y1_i + 1, dtype=np.float32) + 0.5,
        )
        w0 = ((by - qy) * (px - qx) + (qx - bx) * (py - qy)) / den
        w1 = ((qy - ay) * (px - qx) + (ax - qx) * (py - qy)) / den
        w2 = 1.0 - w0 - w1
        inside = (w0 >= -1e-5) & (w1 >= -1e-5) & (w2 >= -1e-5)
        if not np.any(inside):
            continue
        z = (w0 * sz[i, 0] + w1 * sz[i, 1] + w2 * sz[i, 2]).astype(np.float32)
        win = zbuf[y0_i:y1_i + 1, x0_i:x1_i + 1]
        hit = inside & (z < win)
        if not np.any(hit):
            continue
        win[hit] = z[hit]
        findices[y0_i:y1_i + 1, x0_i:x1_i + 1][hit] = int(i) + 1

    hy, hx = np.nonzero(findices > 0)
    if len(hy) > 0:
        f = findices[hy, hx] - 1
        cx = hx.astype(np.float32) + 0.5
        cy = hy.astype(np.float32) + 0.5
        ax, ay = sx[f, 0], sy[f, 0]
        bx, by = sx[f, 1], sy[f, 1]
        qx, qy = sx[f, 2], sy[f, 2]
        den = (by - qy) * (ax - qx) + (qx - bx) * (ay - qy)
        den = np.where(np.abs(den) < 1e-10, 1e-10, den)
        b0 = ((by - qy) * (cx - qx) + (qx - bx) * (cy - qy)) / den
        b1 = ((qy - ay) * (cx - qx) + (ax - qx) * (cy - qy)) / den
        b2 = 1.0 - b0 - b1
        b0 = b0 * inv_w[F[f, 0]]
        b1 = b1 * inv_w[F[f, 1]]
        b2 = b2 * inv_w[F[f, 2]]
        bsum = b0 + b1 + b2
        bsum = np.where(np.abs(bsum) < 1e-12, 1.0, bsum)
        bary[hy, hx, 0] = b0 / bsum
        bary[hy, hx, 1] = b1 / bsum
        bary[hy, hx, 2] = b2 / bsum
    return findices, bary

def _cr_rasterize(pos, tri, resolution, clamp_depth=None, use_depth_prior=0):
    H, W = (resolution, resolution) if isinstance(resolution, int) else (int(resolution[0]), int(resolution[1]))
    findices, bary = _rasterize_numpy(pos[0].detach().cpu().numpy(), tri.detach().cpu().numpy(), H, W)
    dev = pos.device
    return torch.from_numpy(findices).to(device=dev, dtype=torch.int32), torch.from_numpy(bary).to(device=dev, dtype=torch.float32)

def _cr_interpolate(col, findices, barycentric, tri):
    fi = torch.round(findices).long() if findices.is_floating_point() else findices.long()
    f = fi - 1 + (fi == 0).long()
    vcol = col[0, tri.long()[f]]
    result = barycentric.unsqueeze(-1) * vcol
    result = torch.sum(result, dim=-2)
    return result.unsqueeze(0)

cr_mod = types.ModuleType("custom_rasterizer")
cr_mod.rasterize = _cr_rasterize
cr_mod.interpolate = _cr_interpolate
sys.modules["custom_rasterizer"] = cr_mod

def _mesh_vertice_inpaint(texture, mask, vtx_pos, vtx_uv, pos_idx, uv_idx, method="smooth"):
    tex = np.asarray(texture, dtype=np.float32)
    msk = np.asarray(mask, dtype=np.uint8)
    vpos = np.asarray(vtx_pos, dtype=np.float32)
    vuv = np.asarray(vtx_uv, dtype=np.float32)
    pidx = np.asarray(pos_idx, dtype=np.int32)
    uidx = np.asarray(uv_idx, dtype=np.int32)
    H, W, C = tex.shape
    n_v = len(vpos)
    new_tex = tex.copy()
    new_msk = msk.copy()
    if n_v == 0 or len(pidx) == 0:
        return new_tex, new_msk
    uv_flat = uidx.reshape(-1)
    pos_flat = pidx.reshape(-1)
    uv_v = np.clip(np.rint(vuv[uv_flat, 0] * (W - 1)).astype(np.int32), 0, W - 1)
    uv_u = np.clip(np.rint((1.0 - vuv[uv_flat, 1]) * (H - 1)).astype(np.int32), 0, H - 1)
    vtx_mask = np.zeros(n_v, dtype=bool)
    vtx_color = np.zeros((n_v, C), dtype=np.float32)
    valid_hit = msk[uv_u, uv_v] > 0
    if np.any(valid_hit):
        vh_pos = pos_flat[valid_hit]
        vtx_mask[vh_pos] = True
        vtx_color[vh_pos] = tex[uv_u[valid_hit], uv_v[valid_hit]]
    e_src = np.concatenate([pidx[:, 0], pidx[:, 1], pidx[:, 2], pidx[:, 1], pidx[:, 2], pidx[:, 0]])
    e_dst = np.concatenate([pidx[:, 1], pidx[:, 2], pidx[:, 0], pidx[:, 0], pidx[:, 1], pidx[:, 2]])
    dist = np.linalg.norm(vpos[e_src] - vpos[e_dst], axis=1)
    e_w = (1.0 / np.maximum(dist, 1e-4)) ** 2
    for _ in range(120):
        need = (~vtx_mask[e_src]) & vtx_mask[e_dst]
        if not np.any(need):
            break
        u, v, w = e_src[need], e_dst[need], e_w[need]
        w_sum = np.zeros(n_v, dtype=np.float32)
        c_sum = np.zeros((n_v, C), dtype=np.float32)
        np.add.at(w_sum, u, w)
        np.add.at(c_sum, u, vtx_color[v] * w[:, None])
        filled = w_sum > 0.0
        vtx_color[filled] = c_sum[filled] / w_sum[filled, None]
        vtx_mask[filled] = True
    colored_corner = vtx_mask[pos_flat]
    if np.any(colored_corner):
        cu, cv, cp = uv_u[colored_corner], uv_v[colored_corner], pos_flat[colored_corner]
        new_tex[cu, cv] = vtx_color[cp]
        new_msk[cu, cv] = 255
    return new_tex, new_msk

mp_mod = types.ModuleType("hy3dgen.texgen.differentiable_renderer.mesh_processor")
mp_mod.meshVerticeInpaint = _mesh_vertice_inpaint
sys.modules["hy3dgen.texgen.differentiable_renderer.mesh_processor"] = mp_mod

emit("Preparing 3D UV unwrap", 10, "Unwrapping mesh UV islands…")
import trimesh
from hy3dgen.texgen.utils.uv_warp_utils import mesh_uv_wrap
from hy3dgen.texgen.differentiable_renderer.mesh_render import MeshRender

# Patch MeshRender for Apple Silicon MPS float32 safety (avoid float64->mps crash)
_orig_set_mesh = MeshRender.set_mesh
def _safe_set_mesh(self, vtx_pos, pos_idx, vtx_uv=None, uv_idx=None, scale_factor=1.15, auto_center=True):
    vtx_pos = np.asarray(vtx_pos, dtype=np.float32)
    pos_idx = np.asarray(pos_idx, dtype=np.int32)
    if vtx_uv is not None:
        vtx_uv = np.asarray(vtx_uv, dtype=np.float32)
    if uv_idx is not None:
        uv_idx = np.asarray(uv_idx, dtype=np.int32)
    return _orig_set_mesh(self, vtx_pos, pos_idx, vtx_uv=vtx_uv, uv_idx=uv_idx, scale_factor=scale_factor, auto_center=auto_center)
MeshRender.set_mesh = _safe_set_mesh

_orig_set_tex = MeshRender.set_texture
def _safe_set_tex(self, tex):
    if isinstance(tex, np.ndarray):
        tex = Image.fromarray((np.clip(tex, 0.0, 1.0) * 255).astype(np.uint8))
    elif isinstance(tex, torch.Tensor):
        tex = Image.fromarray((np.clip(tex.detach().cpu().numpy(), 0.0, 1.0) * 255).astype(np.uint8))
    tex = tex.resize(self.texture_size).convert("RGB")
    arr = np.asarray(tex, dtype=np.float32) / 255.0
    self.tex = torch.from_numpy(arr).to(device=self.device, dtype=torch.float32)
MeshRender.set_texture = _safe_set_tex

# Also ensure `cosine_similarity` lookat tensor is float32 on MPS
import hy3dgen.texgen.differentiable_renderer.mesh_render as _mr_mod
_orig_bp = MeshRender.back_project
def _safe_back_project(self, image, elev, azim, camera_distance=None, center=None, method=None):
    if isinstance(image, Image.Image):
        image = torch.from_numpy(np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0)
    return _orig_bp(self, image, elev, azim, camera_distance=camera_distance, center=center, method=method)
MeshRender.back_project = _safe_back_project

# Patch `torch.tensor([[0, 0, -1]], device=...)` inside back_project if MPS rejects int64 cosine_similarity
_orig_cos = torch.nn.functional.cosine_similarity
def _safe_cos(x1, x2, dim=1, eps=1e-8):
    if x1.dtype != x2.dtype:
        x1 = x1.to(dtype=x2.dtype)
    return _orig_cos(x1, x2, dim=dim, eps=eps)
torch.nn.functional.cosine_similarity = _safe_cos

# Load input mesh; undo sculptor's +90° X Z-up rotation back to Hunyuan3D canonical Y-up space
mesh = trimesh.load(cfg["mesh_path"], force="mesh")
if isinstance(mesh, trimesh.Scene):
    mesh = mesh.dump(concatenate=True)
if cfg.get("input_is_z_up", True):
    mesh.apply_transform(trimesh.transformations.rotation_matrix(-np.pi / 2.0, [1.0, 0.0, 0.0]))
extra_rot_y_deg = float(cfg.get("rot_y_deg", 0.0))
if abs(extra_rot_y_deg) > 1e-4:
    mesh.apply_transform(trimesh.transformations.rotation_matrix(np.radians(extra_rot_y_deg), [0.0, 1.0, 0.0]))

# Preserve UV unwrap directly on the mesh
mesh = mesh_uv_wrap(mesh)

render_device = "cpu"  # Rasterization runs on NumPy CPU; keeping MeshRender tensors on CPU avoids MPS roundtrips!
cond_size = int(cfg.get("cond_size", 512))
bake_size = int(cfg.get("bake_size", 1024))
tex_size = int(cfg.get("texture_size", 1024))

renderer = MeshRender(default_resolution=cond_size, texture_size=tex_size, device=render_device)
renderer.load_mesh(mesh)

azims = [0, 90, 180, 270, 0, 180]
elevs = [0, 0, 0, 0, 90, -90]
# Give BOTH sides (0°, 180°) AND the dead-on front face (270°: eyes, teeth, open mouth/tongue) full 1.0 bake priority!
view_weights = [1.0, 0.65, 1.0, 1.0, 0.25, 0.25]

emit("Rendering 3D geometry maps", 24, "Projecting 6-view normals & positions…")
normal_maps = [renderer.render_normal(el, az, use_abs_coor=True, return_type="pl") for el, az in zip(elevs, azims)]
position_maps = [renderer.render_position(el, az, return_type="pl") for el, az in zip(elevs, azims)]

debug_dir = cfg.get("debug_dir")
if debug_dir:
    dd = Path(debug_dir)
    dd.mkdir(parents=True, exist_ok=True)
    for idx_v, (nm, pm) in enumerate(zip(normal_maps, position_maps)):
        nm.save(dd / f"normal_{idx_v}_az{azims[idx_v]}_el{elevs[idx_v]}.png")
        pm.save(dd / f"pos_{idx_v}_az{azims[idx_v]}_el{elevs[idx_v]}.png")

if cfg.get("maps_only"):
    emit("Done", 100, "maps_only")
    sys.exit(0)

# Prepare 1..N concept reference views (remove background once if needed, recenter on clean white background)
emit("Preparing concept reference", 32, "Isolating character reference…")
ref_paths = [Path(p) for p in (cfg.get("image_paths") or [cfg["image_path"]])]

def _recenter(im, border_ratio=0.16):
    arr = np.asarray(im.convert("RGBA"))
    nz = np.argwhere(arr[:, :, 3] > 20)
    if nz.size == 0:
        return im.convert("RGB")
    r0, c0 = nz.min(axis=0)
    r1, c1 = nz.max(axis=0)
    crop = im.crop((int(c0), int(r0), int(c1) + 1, int(r1) + 1))
    cw, ch = crop.size
    bw, bh = int(cw * border_ratio), int(ch * border_ratio)
    sq = max(cw + 2 * bw, ch + 2 * bh)
    canvas = Image.new("RGBA", (sq, sq), (255, 255, 255, 0))
    canvas.paste(crop, ((sq - cw) // 2, (sq - ch) // 2))
    return canvas

raw_refs = [Image.open(rp) for rp in ref_paths]
need_rembg = [
    (im.mode != "RGBA" or np.asarray(im)[:, :, 3].min() >= 250)
    for im in raw_refs
]
if any(need_rembg):
    from hy3dgen.rembg import BackgroundRemover
    rembg = BackgroundRemover()
    for idx_r, flag in enumerate(need_rembg):
        if flag:
            raw_refs[idx_r] = rembg(raw_refs[idx_r].convert("RGBA"))
    del rembg
    gc.collect()

ref_ims = [_recenter(im).resize((512, 512)) for im in raw_refs]
if debug_dir:
    for idx_r, r_im in enumerate(ref_ims):
        r_im.save(Path(debug_dir) / f"ref_centered_{idx_r}.png")

# Copy `hunyuanpaint` package into temp workdir and patch hardcoded `'cuda'` / `float64` sites cleanly
patched_pipe_dir = Path(cfg["work_dir"]) / "hunyuanpaint_patched"
shutil.copytree(repo_dir / "hy3dgen" / "texgen" / "hunyuanpaint", patched_pipe_dir, dirs_exist_ok=True)
(patched_pipe_dir / "__init__.py").write_text("", encoding="utf-8")
(patched_pipe_dir / "unet" / "__init__.py").write_text("", encoding="utf-8")
pipe_py = patched_pipe_dir / "pipeline.py"
src = pipe_py.read_text(encoding="utf-8")
src = src.replace(").to('cuda')", ")")
src = src.replace('.half().to("cuda")', ".to(device=device, dtype=self.vae.dtype)")
src = src.replace("np.array(img) / 255.0", "np.array(img, dtype=np.float32) / 255.0")
src = src.replace("torch.range(29, 0, -bsz, device='cuda').long()", "torch.arange(29, -1, -bsz, device=device).long()")
src = src.replace("timesteps = self.solver.ddim_timesteps[index]", "timesteps = self.solver.ddim_timesteps.to(device)[index]")
src = src.replace("self.ddim_alpha_cumprods = torch.from_numpy(self.ddim_alpha_cumprods)", "self.ddim_alpha_cumprods = torch.from_numpy(self.ddim_alpha_cumprods).float()")
src = src.replace("self.ddim_alpha_cumprods_prev = torch.from_numpy(self.ddim_alpha_cumprods_prev)", "self.ddim_alpha_cumprods_prev = torch.from_numpy(self.ddim_alpha_cumprods_prev).float()")
src = src.replace("self.scheduler.set_timesteps(timesteps=timesteps.cpu(), device='cuda')", "self.scheduler.set_timesteps(timesteps=timesteps.cpu(), device=device)")
pipe_py.write_text(src, encoding="utf-8")

# Load UNet2p5DConditionModel directly from `.safetensors` (3.72 GB vs 7.33 GB `.bin`) and instantiate
# `HunyuanPaintPipeline` directly (zero fragile diffusers custom_pipeline reflection)!
emit("Painting 3D surface colors", 45, f"Loading Hunyuan3D-Paint on {device.upper()}…")
from safetensors.torch import load_file as _load_safetensors
from diffusers import AutoencoderKL, DDIMScheduler, LCMScheduler, UNet2DConditionModel
from transformers import CLIPImageProcessor, CLIPTextModel, CLIPTokenizer

ckpt_root = Path(cfg["ckpt_dir"])
unet_dir = ckpt_root / "unet"
# Use checkpoint's exact `unet/modules.py` inside the patched package (with cuda stripped)
if (unet_dir / "modules.py").is_file():
    um_txt = (unet_dir / "modules.py").read_text(encoding="utf-8")
    um_txt = um_txt.replace(".to('cuda')", "").replace('.to("cuda")', "").replace(".cuda()", "")
    (patched_pipe_dir / "unet" / "modules.py").write_text(um_txt, encoding="utf-8")

sys.path.insert(0, str(cfg["work_dir"]))
from hunyuanpaint_patched.unet.modules import UNet2p5DConditionModel
from hunyuanpaint_patched.pipeline import HunyuanPaintPipeline

with open(unet_dir / "config.json", "r", encoding="utf-8") as _f:
    _unet_cfg = json.load(_f)
_base_unet = UNet2DConditionModel(**_unet_cfg)
unet = UNet2p5DConditionModel(_base_unet)
st_path = unet_dir / "diffusion_pytorch_model.safetensors"
bin_path = unet_dir / "diffusion_pytorch_model.bin"
if st_path.is_file():
    _unet_sd = _load_safetensors(str(st_path), device="cpu")
else:
    _unet_sd = torch.load(str(bin_path), map_location="cpu", weights_only=True)
_load_res = unet.load_state_dict(_unet_sd, strict=False)
if _load_res.missing_keys:
    raise RuntimeError(f"Missing UNet weights: {_load_res.missing_keys[:8]}")
del _unet_sd
unet = unet.to(dtype)

vae = AutoencoderKL.from_pretrained(str(ckpt_root / "vae"), torch_dtype=dtype, local_files_only=True)
text_encoder = CLIPTextModel.from_pretrained(str(ckpt_root / "text_encoder"), torch_dtype=dtype, local_files_only=True)
tokenizer = CLIPTokenizer.from_pretrained(str(ckpt_root / "tokenizer"), local_files_only=True)
feature_extractor = CLIPImageProcessor.from_pretrained(str(ckpt_root / "feature_extractor"), local_files_only=True)
base_sched = DDIMScheduler.from_pretrained(str(ckpt_root / "scheduler"), local_files_only=True)

pipe = HunyuanPaintPipeline(
    vae=vae,
    text_encoder=text_encoder,
    tokenizer=tokenizer,
    unet=unet,
    scheduler=base_sched,
    feature_extractor=feature_extractor,
)
pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config, timestep_spacing="trailing")
pipe.set_turbo(True)
pipe.set_progress_bar_config(disable=True)
pipe = pipe.to(device)
if hasattr(pipe, "solver"):
    pipe.solver = pipe.solver.to(device)

camera_info = [
    (((az // 30) + 9) % 12) // {-20: 1, 0: 1, 20: 1, -90: 3, 90: 3}[el]
    + {-20: 0, 0: 12, 20: 24, -90: 36, 90: 40}[el]
    for az, el in zip(azims, elevs)
]

emit("Painting 3D surface colors", 62, f"Generating 6 geometry-locked painted views ({len(ref_ims)} ref{'s' if len(ref_ims) != 1 else ''})…")
gen = torch.Generator(device="cpu").manual_seed(int(cfg.get("seed", 0)))
with torch.inference_mode():
    mvd_images = pipe(
        ref_ims,
        num_inference_steps=30,
        width=512,
        height=512,
        num_in_batch=len(azims),
        camera_info_gen=[camera_info],
        camera_info_ref=[list(range(len(ref_ims)))],
        normal_imgs=[[im.resize((512, 512)) for im in normal_maps]],
        position_imgs=[[im.resize((512, 512)) for im in position_maps]],
        generator=gen,
    ).images

del pipe
gc.collect()
if device == "mps" and hasattr(torch, "mps"):
    torch.mps.empty_cache()
elif device == "cuda":
    torch.cuda.empty_cache()

if debug_dir:
    for idx_v, im in enumerate(mvd_images):
        im.save(Path(debug_dir) / f"painted_{idx_v}_az{azims[idx_v]}_el{elevs[idx_v]}.png")

emit("Baking 3D UV texture", 82, "Projecting 6 views onto 3D UV atlas…")
renderer.set_default_render_resolution(bake_size)
renderer.bake_unreliable_kernel_size = int((2 / 512) * bake_size)

proj_textures = []
proj_weights = []
for view_im, el, az, w in zip(mvd_images, elevs, azims, view_weights):
    view_hi = view_im.convert("RGB").resize((bake_size, bake_size), Image.Resampling.LANCZOS)
    p_tex, p_cos, _ = renderer.back_project(view_hi, el, az)
    proj_textures.append(p_tex)
    proj_weights.append(w * (p_cos ** 4))

texture, trust_mask = renderer.fast_bake_texture(proj_textures, proj_weights)
mask_np = (trust_mask.squeeze(-1).cpu().numpy() * 255).astype(np.uint8)

emit("Baking 3D UV texture", 92, "Inpainting UV seams & saving GLB…")
texture_np = renderer.uv_inpaint(texture, mask_np)
renderer.set_texture(torch.from_numpy(texture_np.astype(np.float32) / 255.0))
textured_mesh = renderer.save_mesh()

# Restore exact original orientation (Z-up for 3D printing)
if abs(extra_rot_y_deg) > 1e-4:
    textured_mesh.apply_transform(trimesh.transformations.rotation_matrix(-np.radians(extra_rot_y_deg), [0.0, 1.0, 0.0]))
if cfg.get("input_is_z_up", True):
    textured_mesh.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2.0, [1.0, 0.0, 0.0]))

out_path = Path(cfg["out_path"])
out_path.parent.mkdir(parents=True, exist_ok=True)
textured_mesh.export(str(out_path))
if debug_dir:
    Image.fromarray(texture_np).save(Path(debug_dir) / "baked_uv_texture.png")
emit("Done", 100, str(out_path))
'''


def _kill_proc(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return
    if os.name == "posix":
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            proc.wait(timeout=2)
            return
        except Exception:  # noqa: BLE001
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                return
            except Exception:  # noqa: BLE001
                pass
    try:
        proc.terminate()
        proc.wait(timeout=2)
    except Exception:  # noqa: BLE001
        try:
            proc.kill()
        except Exception:  # noqa: BLE001
            pass


def _extract_ref_panels(image_path: Path, out_dir: Path) -> list[Path]:
    """Extract 1 or 2 clean character reference views from a 3-panel turnaround, 2x2 sheet, or single image.

    Returns `[side_hero, front_face]` for turnarounds so `HunyuanPaintPipeline` sees BOTH the sweeping side
    silhouette/belly AND the dead-on front face (eyes, teeth, open red mouth/tongue) simultaneously!
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    with Image.open(image_path) as im:
        w, h = im.size
        # 3-panel horizontal turnaround strip (front | side/3-4 | back): feed BOTH middle (side) + left (front)!
        if w >= int(h * 1.65):
            third = w // 3
            p_side = out_dir / "ref_0_side.png"
            p_front = out_dir / "ref_1_front.png"
            im.crop((third, 0, third * 2, h)).save(p_side, format="PNG")
            im.crop((0, 0, third, h)).save(p_front, format="PNG")
            return [p_side, p_front]
        # 2x2 quad sheet with dark divider lines: grab TL 3/4 hero + TR front view
        if w >= 1000 and h >= 700 and (1.05 <= (w / max(1, h)) <= 1.65):
            arr = np.asarray(im.convert("RGB"))
            mid_row = arr[h // 2 - 2:h // 2 + 2].mean()
            mid_col = arr[:, w // 2 - 2:w // 2 + 2].mean()
            if mid_row < 70 and mid_col < 70:
                hw, hh = w // 2, h // 2
                pad_top = min(36, hh // 10)
                p_hero = out_dir / "ref_0_hero.png"
                p_front = out_dir / "ref_1_front.png"
                im.crop((4, pad_top, hw - 4, hh - 4)).save(p_hero, format="PNG")
                im.crop((hw + 4, pad_top, w - 4, hh - 4)).save(p_front, format="PNG")
                return [p_hero, p_front]
        p_single = out_dir / "ref_0_single.png"
        im.save(p_single, format="PNG")
        return [p_single]


def _extract_hero_panel(image_path: Path, out_path: Path) -> Path:
    """Backward-compatible helper returning the primary hero panel."""
    panels = _extract_ref_panels(image_path, out_path.parent)
    if panels[0] != out_path:
        Image.open(panels[0]).save(out_path, format="PNG")
    return out_path


def paint_mesh_from_image(
    mesh_path: Path,
    image_path: Path,
    *,
    input_is_z_up: bool = True,
    rot_y_deg: float = 0.0,
    seed: int = 0,
    bake_size: int = 1024,
    texture_size: int = 1024,
    debug_dir: Path | None = None,
    on_progress: Callable[[str], None] | None = None,
    root: Path | None = None,
) -> bytes:
    """Paint a sculpted `.glb` mesh in 3D using local `Hunyuan3D-Paint` and return textured `.glb` bytes."""
    r = root or ml_root()
    if not tex_painter_ready(r):
        try:
            installer.ensure_ready("tex_painter", on_progress=on_progress, root=root)
        except Exception as exc:  # noqa: BLE001
            raise TexPainterError(str(exc)) from exc
    if not mesh_path.is_file():
        raise TexPainterError(f"3D mesh not found: {mesh_path}")
    if not image_path.is_file():
        raise TexPainterError(f"reference image not found: {image_path}")

    py = sculptor_python(r)
    hw = detect_hardware()
    repo_dir = r / "Hunyuan3D-2"
    ckpt_dir = tex_painter_dir(r)

    with gpu_job("sculpting", "Painting 3D surface colors", "Unwrapping UV atlas…"):
        with tempfile.TemporaryDirectory(prefix="pa_texpaint_") as td:
            td_path = Path(td)
            ref_imgs = _extract_ref_panels(image_path, td_path / "refs")
            out_glb = td_path / "painted.glb"
            worker = td_path / "worker.py"
            worker.write_text(_WORKER_SCRIPT, encoding="utf-8")

            cfg = {
                "repo_dir": str(repo_dir),
                "ckpt_dir": str(ckpt_dir),
                "work_dir": str(td_path),
                "accelerator": hw.accelerator,
                "mesh_path": str(mesh_path),
                "image_path": str(ref_imgs[0]),
                "image_paths": [str(p) for p in ref_imgs],
                "out_path": str(out_glb),
                "input_is_z_up": bool(input_is_z_up),
                "rot_y_deg": float(rot_y_deg),
                "seed": int(seed),
                "cond_size": 512,
                "bake_size": int(bake_size),
                "texture_size": int(texture_size),
                "debug_dir": str(debug_dir) if debug_dir is not None else "",
            }
            env = os.environ.copy()
            env.setdefault("U2NET_HOME", str(r / "models" / "rembg"))
            env.setdefault("HF_HUB_OFFLINE", "1")
            popen_kw: dict = {
                "stdout": subprocess.PIPE,
                "stderr": subprocess.STDOUT,
                "text": True,
                "encoding": "utf-8",
                "errors": "replace",
                "env": env,
            }
            if sys.platform == "win32":
                popen_kw["creationflags"] = getattr(subprocess, "CREATE_NO_WINDOW", 0)
            else:
                popen_kw["start_new_session"] = True

            proc = subprocess.Popen([str(py), str(worker), json.dumps(cfg)], **popen_kw)
            tail: list[str] = []
            try:
                assert proc.stdout is not None
                for line in proc.stdout:
                    s = line.strip()
                    if not s:
                        continue
                    if s.startswith("{"):
                        try:
                            ev = json.loads(s)
                            if ev.get("stage"):
                                ml_job_update(str(ev["stage"]), int(ev.get("pct") or 0), str(ev.get("detail") or ""))
                                if on_progress:
                                    on_progress(f"{ev['stage']}: {ev.get('detail', '')}".strip(": "))
                            continue
                        except json.JSONDecodeError:
                            pass
                    tail.append(s)
                    del tail[:-12]
                rc = proc.wait(timeout=1200)
            except Exception:
                _kill_proc(proc)
                raise
            finally:
                _kill_proc(proc)

            if rc != 0 or not out_glb.is_file():
                raise TexPainterError(f"Local 3D texture painter failed (exit {rc}): {' | '.join(tail[-4:])}")
            ml_job_update("Done", 100, "3D surface paint baked")
            return out_glb.read_bytes()
