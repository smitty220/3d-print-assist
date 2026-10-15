"""The stress analysis as a background job with a live log, and its results for the viewer (surface contours)."""

from __future__ import annotations

import json
import threading
import time
import uuid
from pathlib import Path
from dataclasses import dataclass, field

import numpy as np

from .analysis import PRESETS, AnalysisError, CaseResult, analyse, mesh_metrics, mesh_part
from .controls import MeshSetup, propose
from .mesh import Mesh

KSI = 6.894757
ORDER = ["coarse", "medium", "fine"]
CONVERGENCE_SCALE = 1.5  # the convergence check re-solves with every element size times this


@dataclass
class FemJob:
    id: str
    preset: str
    convergence: bool
    bearing: str = "cosine"  # cosine | contact
    peaks: bool = False  # converge the peaks: refine locally at each hot spot until it stops moving (fem/converge.py)
    peaks_contact: bool = False  # a contact run's rounds solve with contact too (else cosine, times contact / cosine)
    status: str = "running"  # running | done | failed
    log: list[dict] = field(default_factory=list)
    cases: list[dict] = field(default_factory=list)  # summary per load case
    stats: dict = field(default_factory=dict)
    design_key: str = ""
    mode: str = "analyse"  # analyse | size | mesh | plan
    progress: dict = field(default_factory=dict)  # live solver status for the status window
    mesh_stats: dict | None = None  # metrics of the mesh used (or previewed)
    sizing: dict | None = None  # {target, before_mass, after_mass, trials, feasible, can_revert}
    error: str | None = None
    started: float = field(default_factory=time.time)
    finished: float | None = None

    def say(self, kind: str, text: str) -> None:
        self.log.append({"kind": kind, "text": text, "t": round(time.time() - self.started, 1)})

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items() if k not in ("before", "cancel")}

    def monitor(self, stage: str, **info) -> None:
        """Solver progress (fem/solve.py monitor) into `progress`: the current stage, residual history of the running
        solve, and a line per finished stage."""
        p = self.progress
        now = round(time.time() - self.started, 1)
        if stage != p.get("stage"):
            p.setdefault("stages", []).append({"name": stage, "t": now, "case": info.get("case"), "run": p.get("run", "")})
        p["stage"] = stage
        if stage == "iterate":
            if info.get("iteration") == 0:
                p["residuals"] = []
            else:
                p.setdefault("residuals", []).append(info["residual"])
        elif stage == "contact":
            if info.get("pass") == 1:
                p["contact"] = []
            p.setdefault("contact", []).append({k: info.get(k) for k in ("pass", "active", "changed", "iterations")})
        elif stage == "stress" and p.get("residuals"):
            p.setdefault("solves", []).append({"run": p.get("run", ""), "case": info.get("case"), "iterations": info.get("iterations"),
                                               "residual": p["residuals"][-1], "rtol": p.get("rtol")})
        p.update({k: v for k, v in info.items() if k in ("case", "dof", "rtol", "levels", "iteration")})
        p["t"] = now

    def run_label(self, text: str) -> None:
        self.progress["run"] = text


def _case_summary(r: CaseResult, mat) -> dict:
    return {
        "id": r.id, "label": r.label, "category": r.category, "factor": r.factor,
        "vm_max": r.vm_max, "p1_max": r.p1_max, "at": r.at, "u_max": r.u_max,
        "ms_yield": r.ms_yield, "ms_ultimate": r.ms_ultimate, "residual": r.residual,
        "bearing": r.bearing, "contact": r.contact, "calibration": r.calibration,
        "pins": [{"id": p.id, "role": p.role, "force": p.force.round(3).tolist()} for p in r.pins],
        "regions": [{"name": g.name, "vm": g.vm, "p1": g.p1, "at": g.at,
                     "ms_ultimate": mat.ftu / (g.vm * (r.factor if r.category == "limit" else 1.0)) - 1} for g in r.regions],
    }


def surface(mesh: Mesh) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Boundary as linear triangles for display: each 6-node triangle split in four. Returns (node ids, positions, tris)."""
    t = mesh.tris
    lin = np.concatenate([t[:, [0, 3, 5]], t[:, [3, 1, 4]], t[:, [5, 4, 2]], t[:, [3, 4, 5]]])
    ids = np.unique(lin)
    remap = np.full(len(mesh.nodes), -1, np.int64)
    remap[ids] = np.arange(len(ids))
    return ids, mesh.nodes[ids], remap[lin]


def element_edges(mesh: Mesh) -> np.ndarray:
    """The element edges on the surface as segment pairs in `surface()` numbering: each 6-node triangle's three sides,
    corner - mid-side - corner (the split into four display triangles adds no edges of its own)."""
    ids, _pos, _tris = surface(mesh)
    remap = np.full(len(mesh.nodes), -1, np.int64)
    remap[ids] = np.arange(len(ids))
    t = mesh.tris
    seg = np.concatenate([t[:, [0, 3]], t[:, [3, 1]], t[:, [1, 4]], t[:, [4, 2]], t[:, [2, 5]], t[:, [5, 0]]])
    seg = np.unique(np.sort(remap[seg], axis=1), axis=0)
    return seg


def mesh_view(mesh: Mesh) -> dict:
    _ids, pos, tris = surface(mesh)
    return {"positions": pos.astype(np.float32), "tris": tris.astype(np.int32), "edges": element_edges(mesh).astype(np.int32)}


class FemRunner:
    """One analysis at a time. The last finished job and its contour fields are saved under <project>/fem/, so they
    survive a restart (as topology results do)."""

    def __init__(self) -> None:
        self.job: FemJob | None = None  # the latest job of any kind (its log and progress)
        self.results_job: FemJob | None = None  # the latest analysis or sizing job (the results shown)
        self.mesh: Mesh | None = None
        self.results: list[CaseResult] = []
        self._fields: dict | None = None  # loaded from disk: {"positions", "tris", "values": {case: {qty: array}}}
        self._thread: threading.Thread | None = None
        self._folder: Path | None = None
        self.preview: dict | None = None  # {"key", "mesh" (or None once reloaded), "view", "metrics"}: the checked mesh

    def attach(self, folder: Path) -> None:
        """Load the last saved job for a project (once per project)."""
        if self._folder == folder:
            return
        self._folder = folder
        self.job, self.results_job, self.mesh, self.results, self._fields, self.preview = None, None, None, [], None, None
        try:
            z = np.load(folder / "mesh.npz")
            meta = json.loads((folder / "mesh.json").read_text())
            self.preview = {"key": meta["key"], "mesh": None, "metrics": meta["metrics"],
                            "view": {"positions": z["positions"], "tris": z["tris"], "edges": z["edges"]}}
        except (OSError, ValueError, KeyError):
            pass
        try:
            d = json.loads((folder / "job.json").read_text())
            z = np.load(folder / "fields.npz")
        except (OSError, ValueError):
            return
        job = FemJob(**{k: v for k, v in d.items() if k in FemJob.__dataclass_fields__})
        self.job = self.results_job = job
        vals: dict = {}
        disp: dict = {}
        for key in z.files:
            if key.startswith("v__"):
                _v, case, qty = key.split("__", 2)
                vals.setdefault(case, {})[qty] = z[key]
            elif key.startswith("d__"):
                disp[key[3:]] = z[key]
        self._fields = {"positions": z["positions"], "tris": z["tris"], "values": vals, "disp": disp,
                        "edges": z["edges"] if "edges" in z.files else None}

    def _save(self, project) -> None:
        if self.results_job is None or self.mesh is None:
            return
        folder = project.root / "fem"
        folder.mkdir(parents=True, exist_ok=True)
        ids, pos, tris = surface(self.mesh)
        arrays = {"positions": pos.astype(np.float32), "tris": tris.astype(np.int32), "edges": element_edges(self.mesh).astype(np.int32)}
        for r in self.results:
            for q, v in (("vm", r.vm_nodes), ("p1", r.p1_nodes), ("u", r.u_nodes)):
                arrays[f"v__{r.id}__{q}"] = v[ids].astype(np.float32)
            arrays[f"d__{r.id}"] = r.disp[ids].astype(np.float32)
        np.savez_compressed(folder / "fields.npz", **arrays)
        (folder / "job.json").write_text(json.dumps(self.results_job.to_dict(), default=float))
        self._folder = folder
        self._fields = None

    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    @staticmethod
    def mesh_key(project) -> str:
        """The mesh a solve would use: the controls plus the part they mesh."""
        d = project.data.design
        return project.data.mesh.key() + d.sketch_key() + str(len(d.features))

    @staticmethod
    def setup_of(project, build, preset: str = "medium") -> MeshSetup:
        """The project's mesh controls; the best-practice rules (saved, so the user sees them) when there are none."""
        if project.data.mesh.source == "none":
            project.data.mesh = propose(project.data.intent, build.world, preset if preset in PRESETS else "medium",
                                        project.data.mesh.practices)
            project.save()
        return project.data.mesh

    def _save_preview(self, project, key: str, mesh: Mesh, metrics: dict) -> None:
        view = mesh_view(mesh)
        self.preview = {"key": key, "mesh": mesh, "metrics": metrics, "view": view}
        folder = project.root / "fem"
        folder.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(folder / "mesh.npz", **view)
        (folder / "mesh.json").write_text(json.dumps({"key": key, "metrics": metrics}))

    def mesh_payload(self, project) -> dict | None:
        if self.preview is None:
            return None
        v = self.preview["view"]
        return {"positions": v["positions"].round(3).ravel().tolist(), "indices": v["tris"].ravel().tolist(),
                "edges": v["edges"].ravel().tolist(), "metrics": self.preview["metrics"],
                "stale": self.preview["key"] != self.mesh_key(project)}

    def start(self, project, preset: str = "medium", convergence: bool = False, bearing: str = "cosine", peaks: bool = False,
              peaks_contact: bool = False) -> FemJob:
        if self.running():
            raise RuntimeError("An analysis is already running")
        if preset not in PRESETS:
            raise ValueError(f"Unknown mesh preset {preset!r}")
        if bearing not in ("cosine", "contact"):
            raise ValueError(f"Unknown bearing model {bearing!r}")
        self.job = self.results_job = FemJob(id=uuid.uuid4().hex[:8], preset=preset, convergence=convergence, bearing=bearing,
                                             peaks=peaks, peaks_contact=peaks_contact and bearing == "contact")
        self._thread = threading.Thread(target=self._run, args=(self.job, project), daemon=True)
        self._thread.start()
        return self.job

    def start_mesh(self, project) -> FemJob:
        """Mesh the part with its controls (no solve) so the user can check it."""
        if self.running():
            raise RuntimeError("An analysis is already running")
        self.job = FemJob(id=uuid.uuid4().hex[:8], preset="", convergence=False, mode="mesh")
        self._thread = threading.Thread(target=self._mesh, args=(self.job, project), daemon=True)
        self._thread.start()
        return self.job

    def start_plan(self, project, client, model: str, instruction: str = "", max_meshes: int = 3) -> FemJob:
        """Claude proposes mesh controls from the best practices, meshing and checking each proposal."""
        if self.running():
            raise RuntimeError("An analysis is already running")
        self.job = FemJob(id=uuid.uuid4().hex[:8], preset="", convergence=False, mode="plan")
        self.job.cancel = threading.Event()
        self._thread = threading.Thread(target=self._plan, args=(self.job, project, client, model, instruction, max_meshes), daemon=True)
        self._thread.start()
        return self.job

    def make_mesh(self, job: FemJob, project, build, setup: MeshSetup | None = None) -> tuple[Mesh, dict]:
        """Mesh with `setup` (default: the project's controls), log it, keep it as the preview."""
        setup = setup or self.setup_of(project, build)
        job.monitor("mesh")
        t0 = time.time()
        mesh = mesh_part(build.world, setup, log=lambda t: job.say("mesh", t))
        metrics = mesh_metrics(mesh, project.data.intent)
        q = metrics["quality"]
        job.say("mesh", f"{metrics['tets']:,} quadratic tetrahedra, {metrics['dof']:,} DOF (about {metrics['gb']:.1f} GB to solve) "
                        f"in {time.time() - t0:.1f} s; element quality min {q['min']:.2f}, mean {q['mean']:.2f}, "
                        f"{q['below_0_2']} below 0.2")
        for b in metrics["bores"]:
            job.say("mesh", f"{b['id']}: {b['around']} elements around the bore, about {b['through']} through the wall")
        for n in metrics["notes"]:
            job.say("mesh", n)
        job.mesh_stats = metrics
        if setup is project.data.mesh:
            self._save_preview(project, self.mesh_key(project), mesh, metrics)
        return mesh, metrics

    def _mesh(self, job: FemJob, project) -> None:
        try:
            build = project.build()
            if build.world is None:
                raise AnalysisError("There is no part to mesh yet")
            self.make_mesh(job, project, build)
            job.status = "done"
        except AnalysisError as exc:
            job.error, job.status = str(exc), "failed"
            job.say("error", str(exc))
        except Exception as exc:  # noqa: BLE001
            job.error, job.status = f"{type(exc).__name__}: {exc}", "failed"
            job.say("error", job.error)
        finally:
            job.progress["stage"] = "done" if job.status == "done" else "failed"
            job.finished = time.time()

    def _plan(self, job: FemJob, project, client, model: str, instruction: str, max_meshes: int) -> None:
        from ..ai.mesh_plan import run_plan

        try:
            build = project.build()
            if build.world is None:
                raise AnalysisError("There is no part to mesh yet")
            run_plan(job, project, build, client, model, instruction, max_meshes, self)
            job.status = "done"
        except AnalysisError as exc:
            job.error, job.status = str(exc), "failed"
            job.say("error", str(exc))
        except Exception as exc:  # noqa: BLE001
            job.error, job.status = f"{type(exc).__name__}: {exc}", "failed"
            job.say("error", job.error)
        finally:
            job.progress["stage"] = "done" if job.status == "done" else "failed"
            job.finished = time.time()

    def start_sizing(self, project, target_ms: float = 0.10, max_evals: int = 20) -> FemJob:
        if self.running():
            raise RuntimeError("An analysis is already running")
        self.job = self.results_job = FemJob(id=uuid.uuid4().hex[:8], preset="coarse", convergence=False, mode="size")
        self.job.cancel = threading.Event()
        self._thread = threading.Thread(target=self._size, args=(self.job, project, target_ms, max_evals), daemon=True)
        self._thread.start()
        return self.job

    def cancel(self) -> None:
        if self.running() and getattr(self.job, "cancel", None) is not None:
            self.job.cancel.set()

    def _size(self, job: FemJob, project, target_ms: float, max_evals: int) -> None:
        from .sizing import LB, size_part

        try:
            before = project.data.design.model_copy(deep=True)
            pins = {o.id: (o.frame.origin[0], o.frame.origin[1]) for o in project.data.intent.objects if o.kind == "bore"}
            m0 = project.build().mass
            best, trials = size_part(project, pins, target_ms, max_evals, log=job.say, cancel=job.cancel)
            feasible = bool(best and best.feasible)
            job.sizing = {"target": target_ms, "before_mass": m0, "after_mass": best.mass if best else None,
                          "trials": len(trials), "feasible": feasible, "can_revert": False}
            if best is not None and best.design is not None and (feasible or best.mass < float("inf")):
                (project.root / "fem").mkdir(parents=True, exist_ok=True)
                (project.root / "fem" / "before-sizing.json").write_text(before.model_dump_json())  # survives a restart
                project.data.design = best.design
                project.save()
                project.adopt_build(best.build)
                job.before = before
                job.sizing["can_revert"] = True
                job.say("result", f"Kept: {best.mass / LB:.3f} lb (was {m0 / LB:.3f} lb), MS ultimate {best.ms_ult:+.2f} at "
                                  f"{best.governing}{'' if feasible else ' (does not meet the target)'}")
                # Results and contours for the kept part.
                intent = project.data.intent
                mat = intent.rules.material_props()
                mesh, results, stats = analyse(intent, best.build.world, project.shape, "coarse")
                job.stats = stats
                job.cases = [_case_summary(r, mat) for r in results]
                job.design_key = project.data.design.sketch_key() + str(len(project.data.design.features))
                self.mesh, self.results = mesh, results
            job.status = "done"
            job.finished = time.time()
            self._save(project)
        except AnalysisError as exc:
            job.error, job.status = str(exc), "failed"
            job.say("error", str(exc))
        except Exception as exc:  # noqa: BLE001
            job.error, job.status = f"{type(exc).__name__}: {exc}", "failed"
            job.say("error", job.error)
        finally:
            job.finished = time.time()

    def revert_sizing(self, project) -> None:
        job = self.results_job
        if job is None or job.mode != "size" or not (job.sizing or {}).get("can_revert") or self.running():
            raise RuntimeError("Nothing to revert")
        before = getattr(job, "before", None)
        if before is None:  # the server restarted since sizing: the backup on disk
            from ..features.model import Design

            try:
                before = Design.model_validate_json((project.root / "fem" / "before-sizing.json").read_text())
            except (OSError, ValueError):
                raise RuntimeError("The design from before sizing was not saved") from None
        project.data.design = before
        project.save()
        job.sizing["can_revert"] = False
        job.say("result", "Reverted to the design before sizing")

    def _run(self, job: FemJob, project) -> None:
        try:
            build = project.build()
            if build.world is None:
                raise AnalysisError("There is no part to analyse yet")
            job.design_key = project.data.design.sketch_key() + str(len(project.data.design.features))
            intent = project.data.intent
            mat = intent.rules.material_props()
            job.say("read", f"{intent.rules.material}: E {mat.E / KSI / 1e3:.1f} Msi, ν {mat.nu}, Ftu {mat.ftu / KSI:.0f} ksi, "
                            f"Fty {mat.fty / KSI:.0f} ksi ({mat.source})")
            setup = self.setup_of(project, build, job.preset)
            job.run_label("Solve")
            pv = self.preview
            if pv is not None and pv["key"] == self.mesh_key(project) and pv["mesh"] is not None:
                mesh = pv["mesh"]
                job.mesh_stats = pv["metrics"]
                job.say("mesh", f"Using the checked mesh: {len(mesh.tets):,} tetrahedra, {mesh.n_dof:,} DOF")
            else:
                mesh, _metrics = self.make_mesh(job, project, build, setup)
            mesh, results, stats = analyse(intent, build.world, project.shape, job.preset, log=job.say, mesh=mesh, monitor=job.monitor,
                                           bearing=job.bearing)
            mat_d = stats["material"]
            job.stats = stats
            job.cases = [_case_summary(r, mat) for r in results]
            for c in job.cases:
                job.say("result", f"{c['id']}: peak von Mises {c['vm_max'] / KSI:.1f} ksi at {c['regions'][0]['name']}; "
                                  f"MS yield {c['ms_yield']:+.2f}, MS ultimate {c['ms_ultimate']:+.2f}")
            if job.convergence:
                other = f"×{CONVERGENCE_SCALE:g} sizes"
                job.say("read", f"Convergence check: solving again with every element size ×{CONVERGENCE_SCALE:g} (a coarser mesh, same layout)")
                job.run_label("Convergence check")
                _m2, res2, st2 = analyse(intent, build.world, project.shape, job.preset, log=job.say,
                                         setup=setup.scaled(CONVERGENCE_SCALE), monitor=job.monitor, bearing=job.bearing)
                for c, r2 in zip(job.cases, res2):
                    by = {g.name: g.vm for g in r2.regions}
                    c["convergence"] = {"preset": other, "dof": st2["dof"],
                                        "regions": {g["name"]: (g["vm"] / by[g["name"]] - 1) if g["name"] in by else None for g in c["regions"]}}
                    worst = max((abs(v) for v in c["convergence"]["regions"].values() if v is not None), default=0)
                    job.say("result", f"{c['id']}: peaks moved at most {worst:.1%} between this mesh and the coarser one ({st2['dof']:,} DOF)")
            if job.peaks:
                self._converge_peaks(job, project, build, setup, mesh, results, mat)
            self.mesh, self.results = mesh, results
            job.stats["material"] = mat_d
            job.status = "done"
            job.finished = time.time()
            self._save(project)
        except AnalysisError as exc:
            job.error, job.status = str(exc), "failed"
            job.say("error", str(exc))
        except Exception as exc:  # noqa: BLE001 - report in the log
            job.error, job.status = f"{type(exc).__name__}: {exc}", "failed"
            job.say("error", job.error)
        finally:
            job.progress["stage"] = "done" if job.status == "done" else "failed"
            job.finished = time.time()

    def _converge_peaks(self, job: FemJob, project, build, setup: MeshSetup, mesh: Mesh, results: list[CaseResult], mat) -> None:
        """Refine at each hot spot until its peak converges (fem/converge.py) and add the converged peaks, with their
        margins, to each case's summary. The contours stay those of the first mesh."""
        from .analysis import region_labels
        from .converge import BAND, TOL, converge

        intent = project.data.intent
        bores = [o for o in intent.objects if o.kind == "bore"]
        rounds_bearing = "contact" if job.bearing == "contact" and job.peaks_contact else "cosine"
        base = {r.id: r.cosine_vm for r in results} if job.bearing == "contact" and rounds_bearing == "cosine" else None
        how = {"cosine": "the cosine model", "contact": "pin contact"}[rounds_bearing]
        job.say("read", f"Peak convergence: refining locally at every hot spot within {BAND:.0%} of the peak until it moves less than "
                        f"{TOL:.0%}, solving {how}" + (" and carrying each spot's contact / cosine factor from this run" if base else ""))
        job.run_label("Peak convergence")

        def run(s: MeshSetup, rnd: int):
            job.run_label(f"Peak convergence, round {rnd}")
            m = mesh_part(build.world, s, log=lambda t: job.say("mesh", t))
            job.say("mesh", f"Round {rnd}: {len(m.tets):,} tetrahedra, {m.n_dof:,} DOF")
            _m, res, _st = analyse(intent, build.world, project.shape, job.preset, mesh=m, monitor=job.monitor, bearing=rounds_bearing,
                                   log=lambda kind, text: job.say(kind, text) if kind == "contact" and "settled" in text else None)
            return m, {r.id: r.vm_nodes for r in res}

        def on_round(rnd: int, spots) -> None:
            job.progress["peaks"] = {"round": rnd, "spots": [sp.to_dict() for sp in spots]}

        spots = converge(setup, mesh, {r.id: r.vm_nodes for r in results}, run, region_of=lambda x: str(region_labels(x, bores)[0]),
                         base_fields=base, log=lambda t: job.say("peaks", t), on_round=on_round)
        for c, r in zip(job.cases, results):
            ult = r.factor if r.category == "limit" else 1.0
            lim = 1.0 if r.category == "limit" else 1 / r.factor
            out = []
            for sp in (x for x in spots if x.case == r.id):
                d = sp.to_dict()
                d["ms_ultimate"] = mat.ftu / (sp.vm * ult) - 1
                d["ms_yield"] = mat.fty / (sp.vm * lim) - 1
                out.append(d)
                word = {"converged": "converged", "singular": "a singularity (refining raises it without limit: it needs a fillet "
                        "or a smoother load path)", "limit": "not converged in the rounds allowed"}[sp.status]
                job.say("result", f"{r.id} {sp.region}: {sp.vm / KSI:.1f} ksi, MS ultimate {d['ms_ultimate']:+.2f}: {word}")
            settled = [d for d in out if d["status"] == "converged"]
            gov = min(settled or out, key=lambda d: d["ms_ultimate"], default=None)
            c["peaks"] = {"bearing": rounds_bearing, "factored": base is not None, "tol": TOL, "spots": out,
                          "governing": gov and {"id": gov["id"], "region": gov["region"], "vm": gov["vm"], "status": gov["status"],
                                                "ms_ultimate": gov["ms_ultimate"], "ms_yield": gov["ms_yield"]}}

    def field(self, case: str, quantity: str) -> dict:
        """Surface contour of one result: positions (world mm), triangles, a value per vertex, and its range."""
        if self.mesh is not None and self.results:
            r = next((x for x in self.results if x.id == case), None)
            if r is None:
                raise KeyError(f"No load case {case!r} in the results")
            values = {"vm": r.vm_nodes, "p1": r.p1_nodes, "u": r.u_nodes}.get(quantity)
            if values is None:
                raise KeyError(f"Unknown quantity {quantity!r}")
            ids, pos, tris = surface(self.mesh)
            v = values[ids]
            d = r.disp[ids]
            edges = element_edges(self.mesh)
        elif self._fields is not None:
            v = self._fields["values"].get(case, {}).get(quantity)
            if v is None:
                raise KeyError(f"No {quantity} field for load case {case!r}")
            pos, tris = self._fields["positions"], self._fields["tris"]
            d = self._fields["disp"].get(case, np.zeros_like(pos))
            edges = self._fields.get("edges")
            if edges is None:  # saved before element edges were: the display triangles' sides
                t = np.asarray(tris)
                edges = np.unique(np.sort(np.concatenate([t[:, [0, 1]], t[:, [1, 2]], t[:, [2, 0]]]), axis=1), axis=0)
        else:
            raise KeyError("No results yet")
        return {"positions": pos.round(3).ravel().tolist(), "indices": tris.ravel().tolist(),
                "values": v.round(4).tolist(), "min": float(v.min()), "max": float(v.max()), "quantity": quantity,
                "displacement": np.asarray(d).round(6).ravel().tolist(), "edges": np.asarray(edges).ravel().tolist(),
                "peak": pos[int(np.argmax(v))].round(3).tolist(), "peak_index": int(np.argmax(v))}
