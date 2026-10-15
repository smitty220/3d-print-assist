"""The open design and its build state: what was last built, what is out of date, and every change made to it.

Every change goes through `change()` (a list of ops, applied together or not at all). In manual mode a change only
marks what it affects as out of date (⚡); `update()` rebuilds. In auto mode a change rebuilds at once. Changes by the
AI are always built and checked before they are kept: if they would make a feature fail, they are refused and the
design is left as it was, so the AI never leaves the model out of date or broken.
"""

from __future__ import annotations

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
        self._mesh: tuple[int, dict] | None = None
        self.update(checkpoint=False)

    @property
    def design(self) -> Design:
        return self.project.data.design

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
            self.project.data.design = trial
            self.project.save()
            self.rev += 1
            self.log(actor, message or ", ".join(op.get("op", "?") for op in ops))
            if actor != design_ops.USER or self.mode == "auto":
                self.update(message=f"{'AI' if actor != design_ops.USER else 'Edit'}: {message or 'change'}")
            return made

    def set_mode(self, mode: str) -> None:
        with self.lock:
            self.project.data.settings.update_mode = mode
            self.project.save()
            if mode == "auto":
                self.update()

    # ---------- building ----------

    def update(self, message: str = "Update", checkpoint: bool = True) -> BuildResult:
        with self.lock:
            result = self.engine.build(self.design, self.project.data.rules, fonts_dir=self.project.root / "fonts")
            self.last = result
            self.built_rev = self.rev
            self.built_hashes = dict(result.hashes)
            self.built_params = dict(result.param_values)
            if result.complete:
                self.last_good = result
                if checkpoint:
                    self.project.checkpoint(message)
            return result

    def pending(self) -> list[str]:
        """Ids of the sketches and features whose inputs changed since the last update."""
        p = plan(self.design, self.project.data.rules)
        return [k for k, h in p.hashes().items() if self.built_hashes.get(k) != h]

    def revert_last_ai(self, actor: str = design_ops.USER) -> str:
        """Put the design back as it was before the last change the AI kept. Returns that change's message."""
        import subprocess

        from ..features.model import Design

        with self.lock:
            hist = self.project.history(200)
            ai = next((h for h in hist if h["message"].startswith("AI:")), None)
            if ai is None:
                raise ChangeRefused("there is no AI change to revert")
            run = subprocess.run(["git", "show", f"{ai['id']}^:project.json"], cwd=self.project.root,
                                 capture_output=True, text=True, encoding="utf-8")
            if run.returncode != 0:
                raise ChangeRefused("could not read the design before that change")
            import json

            self.project.data.design = Design.model_validate(json.loads(run.stdout)["design"])
            self.project.save()
            self.rev += 1
            self.log(actor, f"Reverted: {ai['message']}")
            self.update(message=f"Revert {ai['message']}")
            return ai["message"]

    def render(self, view: str = "iso", size: int = 640) -> bytes:
        from ..geometry.render import render

        with self.lock:
            good = self.last_good
            bodies = []
            if good is not None:
                colors = {b.id: b.color for b in self.design.bodies}
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
                bodies.append({"id": b.id, "name": b.name, "color": b.color, "status": worst, "features": [f["id"] for f in mine]})

            pending = [k for k in hashes if k in stale]
            failed = bool(self.last and not self.last.complete)
            return {
                "name": self.project.data.name, "slug": self.project.slug,
                "rev": self.rev, "built_rev": self.built_rev, "mode": self.mode,
                "pending": pending, "pending_count": len(pending), "stale": bool(pending), "failed": failed,
                "exportable": not pending and not failed and self.last_good is not None,
                "rollback": d.rollback, "params": params, "features": features, "sketches": sketches, "bodies": bodies,
                "errors": r.errors,
            }

    def part(self) -> dict:
        """The last good part, one mesh per body, cached until the next good update."""
        with self.lock:
            good = self.last_good
            key = id(good)
            if self._mesh is None or self._mesh[0] != key:
                bodies = []
                if good is not None:
                    for b in self.design.bodies:
                        shape = good.parts.get(b.id)
                        if shape is None:
                            continue
                        m = mesh_shape(shape, b.id).to_dict()
                        m["face_names"] = good.body_names.get(b.id, [])
                        bodies.append({"id": b.id, "name": b.name, "color": b.color, "mesh": m})
                self._mesh = (key, {"bodies": bodies, "volume": good.volume if good else 0.0,
                                    "mass": good.mass if good else 0.0})
            pending = self.pending()
            return self._mesh[1] | {"rev": self.rev, "built_rev": self.built_rev, "stale": bool(pending),
                                    "failed": bool(self.last and not self.last.complete)}


def _literal(expr: str) -> bool:
    try:
        float(expr)
        return True
    except ValueError:
        return False
