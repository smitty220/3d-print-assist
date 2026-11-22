"""Local 3D Clay Sculptor for Character Mode (Hunyuan3D-2mv / Hunyuan3D-2mini on Apple Silicon MPS, CUDA, or CPU).

Invokes an isolated worker script inside `ml_root()/venv` as a managed subprocess so PyTorch / GPU memory
is completely released as soon as the `.glb` mesh finishes generating.
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

from PIL import Image

from . import gpu_job, ml_job_update, ml_root
from .hardware import detect_hardware
from .installer import installer, sculptor_python, sculptor_ready


class SculptorError(RuntimeError):
    pass


_WORKER_SCRIPT = r'''
import json, os, sys
from pathlib import Path
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

emit("Loading 3D Sculptor weights", 15, f"Using {device.upper()} ({cfg['repo']} / {cfg['subfolder']})")

from hy3dgen.rembg import BackgroundRemover
from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline

rembg = BackgroundRemover()
views_in = cfg["views"]
prepared = {}
for k, vpath in views_in.items():
    im = Image.open(vpath).convert("RGBA")
    im = rembg(im)
    prepared[k] = im

emit("Molding 3D clay geometry", 45, "Running flow-matching shape diffusion…")
dtype = torch.float32 if device == "cpu" else torch.float16
pipe = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
    cfg["repo"],
    subfolder=cfg["subfolder"],
    variant="fp16",
    use_safetensors=True,
    device=device,
    dtype=dtype,
)

is_mv = "mv" in cfg["subfolder"]
inp = prepared if is_mv else (prepared.get("front") or next(iter(prepared.values())))
steps = int(cfg.get("steps", 30))
mesh = pipe(
    image=inp,
    num_inference_steps=steps,
    octree_resolution=int(cfg.get("octree_resolution", 256)),
    num_chunks=20000,
    output_type="trimesh",
)[0]

out_path = Path(cfg["out_path"])
out_path.parent.mkdir(parents=True, exist_ok=True)
emit("Optimizing 3D mesh", 88, "Converting Y-up to Z-up and decimation for crisp 3D printing…")
import numpy as np
import trimesh
# Convert standard glTF Y-up output from Hunyuan3D into Z-up 3D-printing orientation (+90° about X)
mesh.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2.0, [1.0, 0.0, 0.0]))
# Quadric decimation to ~40k faces keeps full organic detail while making viewport & flexi CSG instant
try:
    max_faces = int(cfg.get("max_faces", 40000))
    if len(mesh.faces) > max_faces:
        import pymeshlab
        ms = pymeshlab.MeshSet()
        ms.add_mesh(pymeshlab.Mesh(mesh.vertices, mesh.faces))
        ms.apply_filter("meshing_decimation_quadric_edge_collapse", targetfacenum=max_faces, preserveboundary=True)
        cm = ms.current_mesh()
        if cm.face_number() > 100:
            mesh = trimesh.Trimesh(vertices=cm.vertex_matrix(), faces=cm.face_matrix(), process=False)
except Exception:
    pass
emit("Exporting raw sculpt mesh", 94, out_path.name)
mesh.export(str(out_path))
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


def _split_turnaround(image_path: Path, out_dir: Path, views: list[str]) -> dict[str, str]:
    """If the Design File is a 3-panel Front/Side/Back turnaround sheet, crop each view cleanly."""
    with Image.open(image_path) as im:
        im = im.convert("RGB")
        w, h = im.size
        if len(views) >= 3 and w >= int(h * 1.6):
            third = w // 3
            crops = {
                "front": im.crop((0, 0, third, h)),
                "left": im.crop((third, 0, third * 2, h)),
                "back": im.crop((third * 2, 0, w, h)),
            }
            out: dict[str, str] = {}
            for k, c in crops.items():
                p = out_dir / f"{k}.png"
                c.save(p, format="PNG")
                out[k] = str(p)
            return out
        p = out_dir / "front.png"
        im.save(p, format="PNG")
        return {"front": str(p)}


def sculpt_mesh_from_image(
    image_path: Path,
    views: list[str] | None = None,
    steps: int = 20,
    on_progress: Callable[[str], None] | None = None,
    root: Path | None = None,
) -> bytes:
    """Run local Hunyuan3D-2mv / 2mini on the approved Character Design File and return raw `.glb` bytes."""
    r = root or ml_root()
    if not sculptor_ready(r):
        try:
            installer.ensure_ready("sculptor", on_progress=on_progress, root=root)
        except Exception as exc:  # noqa: BLE001
            raise SculptorError(str(exc)) from exc
    if not image_path.is_file():
        raise SculptorError(f"concept design image not found: {image_path}")
    py = sculptor_python(r)
    hw = detect_hardware()
    repo_dir = r / "Hunyuan3D-2"

    with gpu_job("sculpting", "Molding 3D clay", f"Loading {hw.sculptor_label} on {hw.gpu_name}…"):
        with tempfile.TemporaryDirectory(prefix="pa_sculpt_") as td:
            td_path = Path(td)
            view_files = _split_turnaround(image_path, td_path, views or [])
            out_glb = td_path / "sculpt.glb"
            worker = td_path / "worker.py"
            worker.write_text(_WORKER_SCRIPT, encoding="utf-8")

            cfg = {
                "repo_dir": str(repo_dir),
                "repo": hw.sculptor_repo,
                "subfolder": hw.sculptor_subfolder,
                "accelerator": hw.accelerator,
                "views": view_files,
                "out_path": str(out_glb),
                "steps": max(10, min(int(steps), 50)),
                "octree_resolution": 192 if hw.tier == "high" else 160,
                "max_faces": 40000,
            }
            env = os.environ.copy()
            env["HY3DGEN_MODELS"] = str(r / "models" / "sculptor")
            env.setdefault("U2NET_HOME", str(r / "models" / "rembg"))
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
                    del tail[:-10]
                rc = proc.wait(timeout=1200)
            except Exception:
                _kill_proc(proc)
                raise
            finally:
                _kill_proc(proc)

            if rc != 0 or not out_glb.is_file():
                raise SculptorError(f"Local 3D sculptor failed (exit {rc}): {' | '.join(tail[-3:])}")
            ml_job_update("Done", 100, "3D sculpt ready")
            return out_glb.read_bytes()
