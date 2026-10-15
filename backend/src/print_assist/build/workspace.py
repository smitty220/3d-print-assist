"""The open design and its build state: what was last built, what is out of date, and every change made to it.

The design being edited is a working copy: changes stay unsaved (the title shows •) until `save()` writes it to the
design's project.json and records a version in its git history. A small recovery copy (.recovery.json) is written on
every change, so work survives a crash; `undo()` / `redo()` step through every change, the AI's included.

Every change goes through `change()` (a list of ops, applied together or not at all). In manual mode a change only
marks what it affects as out of date (⚡); `update()` rebuilds. In auto mode a change rebuilds at once. Changes by the
AI are always built and checked before they are kept: if they would make a feature fail, they are refused and the
design is left as it was, so the AI never leaves the model out of date or broken.
"""

from __future__ import annotations

import json
import subprocess
import threading
import time
from dataclasses import dataclass, field

from ..features.design import BuildResult
from ..features.library import FEATURES
from ..features.model import Design
from ..geometry.tessellate import mesh_shape
from ..project import Project
from . import ops as design_ops
from .engine import Engine, plan

FEATURE_LABEL = {k: f.label for k, f in FEATURES.items()}


class ChangeRefused(ValueError):
    pass


RECOVERY = ".recovery.json"
UNDO_LIMIT = 100


@dataclass
class Step:
    """One change on the undo stack: the design before it, and what it was."""

    before: str  # the design as JSON
    label: str
    actor: str


@dataclass
class Activity:
    actor: str
    text: str
    at: float = field(default_factory=time.time)
    ok: bool = True

    def to_dict(self) -> dict:
        return dict(self.__dict__)


class Workspace:
    def __init__(self, project: Project, engine: Engine | None = None):
        self.project = project
        self.engine = engine or Engine()
        self.lock = threading.RLock()
        self.rev = 0  # bumps on every change to the design
        self.built_rev = -1  # the rev the last update built
        self.last: BuildResult | None = None  # the last update (statuses)
        self.last_good: BuildResult | None = None  # the last update that built completely (what is shown)
        self.built_hashes: dict[str, str] = {}
        self.built_params: dict = {}
        self.activity: list[Activity] = []
        self._mesh: tuple[int, dict[str, dict]] | None = None
        self._checks: tuple[tuple, dict] | None = None  # the print checks, for one build, palette and printer
        self._checks_lock = threading.Lock()
        self.working: Design = project.data.design.model_copy(deep=True)
        self.saved_json = self.working.model_dump_json()
        self.undo_stack: list[Step] = []
        self.redo_stack: list[Step] = []
        rec = project.root / RECOVERY
        self.recovery = rec.exists() and rec.read_text(encoding="utf-8") != self.saved_json
        self.update()

    @property
    def design(self) -> Design:
        return self.working

    @property
    def dirty(self) -> bool:
        return self.working.model_dump_json() != self.saved_json

    def file_state(self) -> dict:
        return {"open": True, "slug": self.project.slug, "name": self.project.data.name, "dirty": self.dirty,
                "recovery": self.recovery,
                "undo": self.undo_stack[-1].label if self.undo_stack else None,
                "redo": self.redo_stack[-1].label if self.redo_stack else None}

    def _set(self, design: Design) -> None:
        """Make `design` the working copy (caller holds the lock) and keep the recovery copy current."""
        self.working = design
        self.rev += 1
        rec = self.project.root / RECOVERY
        if self.dirty:
            rec.write_text(self.working.model_dump_json(), encoding="utf-8")
        else:
            rec.unlink(missing_ok=True)

    @property
    def mode(self) -> str:
        return self.project.data.settings.update_mode

    def log(self, actor: str, text: str, ok: bool = True) -> None:
        self.activity.append(Activity(actor, text, ok=ok))
        del self.activity[:-200]

    # ---------- changes ----------

    def change(self, ops: list[dict], actor: str = design_ops.USER, message: str = "", base_rev: int | None = None) -> list:
        """Apply ops together. Returns what each op made (new ids). Raises OpError or ChangeRefused, leaving the design
        unchanged."""
        with self.lock:
            if base_rev is not None and base_rev != self.rev:
                raise ChangeRefused(f"the design changed since revision {base_rev} (now {self.rev}); read it again")
            trial = self.design.model_copy(deep=True)
            made = [design_ops.apply(trial, op, actor) for op in ops]
            if actor != design_ops.USER:
                before = {f.id for f in (self.last.features if self.last else []) if f.status == "fail"}
                result = self.engine.build(trial, self.project.data.rules, fonts_dir=self.project.root / "fonts")
                broken = [f for f in result.features if f.status == "fail" and f.id not in before]
                if broken:
                    msg = "; ".join(f"{f.id}: {f.error}" for f in broken)
                    self.log(actor, f"Refused: {message or 'change'} ({msg})", ok=False)
                    raise ChangeRefused(f"not kept, it would make {msg}")
            label = message or ", ".join(op.get("op", "?") for op in ops)
            self.undo_stack.append(Step(self.working.model_dump_json(), label, actor))
            del self.undo_stack[:-UNDO_LIMIT]
            self.redo_stack.clear()
            self._set(trial)
            self.log(actor, label)
            if actor != design_ops.USER or self.mode == "auto":
                self.update()
            return made

    # ---------- undo, save, versions ----------

    def _step(self, src: list[Step], dst: list[Step], verb: str, actor: str) -> str:
        with self.lock:
            if not src:
                raise ChangeRefused(f"nothing to {verb}")
            step = src.pop()
            dst.append(Step(self.working.model_dump_json(), step.label, step.actor))
            self._set(Design.model_validate_json(step.before))
            self.log(actor, f"{verb.capitalize()}: {step.label}")
            if self.mode == "auto":
                self.update()
            return step.label

    def undo(self, actor: str = design_ops.USER) -> str:
        return self._step(self.undo_stack, self.redo_stack, "undo", actor)

    def redo(self, actor: str = design_ops.USER) -> str:
        return self._step(self.redo_stack, self.undo_stack, "redo", actor)

    def save(self, message: str = "") -> bool:
        """Write the working copy to disk and record a version. Returns whether a version was recorded."""
        with self.lock:
            self.project.data.design = self.working.model_copy(deep=True)
            self.project.save()
            self.saved_json = self.working.model_dump_json()
            (self.project.root / RECOVERY).unlink(missing_ok=True)
            self.recovery = False
            try:
                (self.project.root / "thumbnail.png").write_bytes(self.render("iso", 320))
            except Exception:  # noqa: BLE001 - a thumbnail is a nicety
                pass
            self.rev += 1
            committed = self.project.checkpoint(message.strip() or "Save")
            self.log(design_ops.USER, f"Saved{': ' + message if message else ''}")
            return committed

    def revert_to_saved(self) -> None:
        with self.lock:
            self.undo_stack.append(Step(self.working.model_dump_json(), "Revert to saved", design_ops.USER))
            self._set(Design.model_validate_json(self.saved_json))
            self.recovery = False
            if self.mode == "auto":
                self.update()

    def recover(self, keep: bool) -> None:
        """Take the recovery copy as the working copy (keep), or throw it away."""
        with self.lock:
            rec = self.project.root / RECOVERY
            if keep and rec.exists():
                self.undo_stack.append(Step(self.working.model_dump_json(), "Recover unsaved changes", design_ops.USER))
                self._set(Design.model_validate_json(rec.read_text(encoding="utf-8")))
                if self.mode == "auto":
                    self.update()
            elif not keep:
                rec.unlink(missing_ok=True)
            self.recovery = False

    def restore_version(self, commit: str) -> None:
        """Load a saved version into the working copy (unsaved until you save; undo brings back what was there)."""
        with self.lock:
            run = subprocess.run(["git", "show", f"{commit}:project.json"], cwd=self.project.root,
                                 capture_output=True, text=True, encoding="utf-8")
            if run.returncode != 0:
                raise ChangeRefused(f"no saved version {commit!r}")
            self.undo_stack.append(Step(self.working.model_dump_json(), f"Restore version {commit}", design_ops.USER))
            self._set(Design.model_validate(json.loads(run.stdout)["design"]))
            if self.mode == "auto":
                self.update()

    def set_mode(self, mode: str) -> None:
        with self.lock:
            self.project.data.settings.update_mode = mode
            self.project.save()
            if mode == "auto":
                self.update()

    # ---------- building ----------

    def update(self) -> BuildResult:
        with self.lock:
            result = self.engine.build(self.design, self.project.data.rules, fonts_dir=self.project.root / "fonts")
            self.last = result
            self.built_rev = self.rev
            self.built_hashes = dict(result.hashes)
            self.built_params = dict(result.param_values)
            if result.complete:
                self.last_good = result
            return result

    def build_variant(self, variant: str | None) -> tuple[Design, BuildResult]:
        """Build the design with a variant's parameter values (None: as it is), reusing the engine's cache."""
        from ..variants import with_variant

        with self.lock:
            d = self.design
            v = None
            if variant:
                v = d.variant(variant)
                if v is None:
                    raise ValueError(f"no variant {variant!r}")
            vd = with_variant(d, v)
            result = self.engine.build(vd, self.project.data.rules, fonts_dir=self.project.root / "fonts")
            return vd, result

    def pending(self) -> list[str]:
        """Ids of the sketches and features whose inputs changed since the last update."""
        p = plan(self.design, self.project.data.rules)
        return [k for k, h in p.hashes().items() if self.built_hashes.get(k) != h]

    def revert_last_ai(self, actor: str = design_ops.USER) -> str:
        """Undo back to just before the last change an AI made (later changes go onto the redo stack too)."""
        with self.lock:
            if not any(st.actor != design_ops.USER for st in self.undo_stack):
                raise ChangeRefused("there is no AI change to revert")
            while True:
                ai = self.undo_stack[-1].actor != design_ops.USER
                label = self.undo(actor)
                if ai:
                    self.update()
                    return f"AI: {label}"

    def checks(self) -> dict:
        """The print checks of the last good build, worked out once per build (and palette and printer): requests that
        arrive while they run wait and share the result, and always get the newest build's."""
        from ..validate import validate

        with self._checks_lock:
            with self.lock:
                good, design = self.last_good, self.design
                key = (id(good), design.model_dump_json(include={"filaments", "bodies"}), self.project.data.settings.printer,
                       repr(self.project.data.rules))
                stale = bool(self.pending())
            if self._checks is None or self._checks[0] != key:
                rep = validate(good, self.project.data.rules, self.project.printer, design=design).to_dict()
                self._checks = (key, rep)
            return self._checks[1] | {"stale": stale}

    def render(self, view: str = "iso", size: int = 640) -> bytes:
        from ..geometry.render import render

        with self.lock:
            good = self.last_good
            bodies = []
            if good is not None:
                colors = {b.id: self.design.body_color(b) for b in self.design.bodies}
                bodies = [(shape, colors.get(bid)) for bid, shape in good.parts.items()]
        return render(bodies, view, size)

    def faces(self, body: str = "B1") -> list[dict]:
        """The shown part's faces, compactly: what the AI needs to pick one (index, name, type, centre, normal, area)."""
        from build123d import GeomType

        good = self.last_good
        shape = good.parts.get(body) if good else None
        if shape is None:
            return []
        names = good.body_names.get(body, [])
        out = []
        for i, f in enumerate(shape.faces()):
            c = f.center()
            item = {"index": i, "name": names[i] if i < len(names) else None, "type": f.geom_type.name.lower(),
                    "center": [round(c.X, 2), round(c.Y, 2), round(c.Z, 2)], "area": round(f.area, 1)}
            if f.geom_type == GeomType.PLANE:
                n = f.normal_at()
                item["normal"] = [round(n.X, 3), round(n.Y, 3), round(n.Z, 3)]
            out.append(item)
        return out

    # ---------- what the UI and the AI read ----------

    def tree(self) -> dict:
        with self.lock:
            d = self.design
            p = plan(d, self.project.data.rules)
            r = p.resolved
            hashes = p.hashes()
            stale = {k for k, h in hashes.items() if self.built_hashes.get(k) != h}
            last = {f.id: f for f in (self.last.features if self.last else [])}

            params = []
            for prm in d.parameters:
                err = r.errors.get(f"param:{prm.name}")
                val = r.values.get(prm.name)
                status = "fail" if err else ("stale" if self.built_params.get(prm.name) != val else "ok")
                params.append({"name": prm.name, "quantity": prm.quantity, "expr": prm.expr, "value": val,
                               "comment": prm.comment, "relation": prm.quantity != "text" and not _literal(prm.expr),
                               "status": status, "error": err,
                               "users": sorted(r.users.get(prm.name, set())), "inputs": sorted(r.inputs.get(f"param:{prm.name}", set()))})

            features = []
            for i, (fi, fp) in enumerate(zip(d.features, p.features)):
                built = last.get(fi.id)
                if fi.id in stale:
                    status, message = "stale", "inputs changed since the last update"
                elif built is not None:
                    status, message = built.status, built.error or built.message
                else:
                    status, message = "stale", None
                features.append({
                    "id": fi.id, "kind": fi.kind, "label": FEATURE_LABEL.get(fi.kind, fi.kind), "index": i, "body": fi.body,
                    "status": status, "message": message, "error": (built.error if built else None) or fp.error,
                    "locked": fi.locked, "suppressed": fi.suppressed, "params": fp.params, "exprs": fi.exprs,
                    "refs": fi.refs, "edges": [e.model_dump() for e in fi.edges], "faces": [f.model_dump() for f in fi.faces],
                    "inputs": sorted(r.inputs.get(f"feature:{fi.id}", set())),
                    "driven": built.driven if built else {},
                })

            sketches = []
            built_sk = self.last.sketch_status if self.last else {}
            sk_err = self.last.sketch_errors if self.last else {}
            for sk in d.sketches:
                status = "stale" if sk.id in stale else built_sk.get(sk.id, "stale")
                sketches.append({"id": sk.id, "name": sk.name, "status": status,
                                 "error": sk_err.get(sk.id) or p.sketches[sk.id].error,
                                 "placement": sk.placement.model_dump(), "dof": sk.report.dof if sk.report else None,
                                 "used_by": [f.id for f in d.features if sk.id in f.refs.values()],
                                 "inputs": sorted(r.inputs.get(f"sketch:{sk.id}", set()))})

            rank = {"ok": 0, "warn": 1, "skipped": 2, "fail": 3, "stale": 4}
            bodies = []
            for b in d.bodies:
                mine = [f for f in features if f["body"] == b.id]
                worst = max((f["status"] for f in mine), key=lambda s: rank.get(s, 0), default="ok")
                fed = [f["id"] for f in features if b.id in (f["params"].get("into"), f["params"].get("tool")) and f["body"] != b.id]
                bodies.append({"id": b.id, "name": b.name, "color": d.body_color(b), "own_color": b.color,
                               "filament": b.filament, "status": worst, "features": [f["id"] for f in mine], "fed_by": fed})

            pending = [k for k in hashes if k in stale]
            failed = bool(self.last and not self.last.complete)
            return {
                "name": self.project.data.name, "slug": self.project.slug, "dirty": self.dirty,
                "undo": self.undo_stack[-1].label if self.undo_stack else None,
                "redo": self.redo_stack[-1].label if self.redo_stack else None,
                "rev": self.rev, "built_rev": self.built_rev, "mode": self.mode,
                "pending": pending, "pending_count": len(pending), "stale": bool(pending), "failed": failed,
                "exportable": not pending and not failed and self.last_good is not None,
                "rollback": d.rollback, "params": params, "features": features, "sketches": sketches, "bodies": bodies,
                "errors": r.errors, "filaments": [f.model_dump() for f in d.filaments],
                "variants": [v.model_dump() for v in d.variants], "colorways": [c.model_dump() for c in d.colorways],
                "references": [x.model_dump() for x in d.references],
            }

    def part(self) -> dict:
        """The last good part, one mesh per body (meshes cached until the next good update; names and colours current)."""
        with self.lock:
            good = self.last_good
            if self._mesh is None or self._mesh[0] != id(good):
                meshes = {}
                for bid, shape in (good.parts if good else {}).items():
                    m = mesh_shape(shape, bid).to_dict()
                    m["face_names"] = good.body_names.get(bid, [])
                    meshes[bid] = m
                self._mesh = (id(good), meshes)
            meshes = self._mesh[1]
            d = self.design
            bodies = [{"id": b.id, "name": b.name, "color": d.body_color(b), "mesh": meshes[b.id]}
                      for b in d.bodies if b.id in meshes]
            pending = self.pending()
            return {"bodies": bodies, "volume": good.volume if good else 0.0, "mass": good.mass if good else 0.0,
                    "rev": self.rev, "built_rev": self.built_rev, "stale": bool(pending),
                    "failed": bool(self.last and not self.last.complete)}


def _literal(expr: str) -> bool:
    try:
        float(expr)
        return True
    except ValueError:
        return False
