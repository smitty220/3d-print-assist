"""The incremental build engine.

Every sketch and feature gets an input hash computed without building anything: a sketch's from its geometry,
constraints (formula values filled in) and placement; a feature's from its kind, resolved parameters, picks, body,
what it refers to, and the hash of the feature before it. So a change marks exactly the features from the first one
it touches to the end of the tree, and an update rebuilds only those: each feature's result (every body's shape after
it, plus the outputs later features read) is cached by its hash and reused while the hash is unchanged.

A feature that fails stops the update: the features after it are skipped (CATIA does the same) and the caller keeps
showing the last good part. Statuses: ok, warn (built, but one of its checks fails), fail, skipped.
"""

from __future__ import annotations

import hashlib
import json
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass, field

from build123d import Compound, Shape

from ..features.context import BuildContext, FeatureError
from ..features.design import BuildResult, FeatureResult, apply_output, attribute
from ..features.library import FEATURES, resolve_params
from ..features.model import Check, Design, FeatureOutput
from ..params.graph import Resolved, resolve
from ..rules import PrintRules
from ..sketch.model import Sketch
from ..sketch.solve import SketchSolveError, solve


def _h(obj) -> str:
    return hashlib.sha1(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()[:16]


@dataclass
class SketchPlan:
    id: str
    hash: str
    sketch: Sketch | None  # formulas filled in; still to be solved when `needs_solve`
    needs_solve: bool = False
    error: str | None = None


@dataclass
class FeaturePlan:
    id: str
    hash: str
    params: dict
    error: str | None = None


@dataclass
class Plan:
    resolved: Resolved
    sketches: dict[str, SketchPlan]
    features: list[FeaturePlan]

    def hashes(self) -> dict[str, str]:
        return {**{s.id: s.hash for s in self.sketches.values()}, **{f.id: f.hash for f in self.features}}


@dataclass
class _Snapshot:
    parts: dict[str, Shape]
    outputs: dict[str, FeatureOutput]
    params: dict[str, dict]
    result: FeatureResult
    checks: list[Check] = field(default_factory=list)


def plan(design: Design, rules: PrintRules) -> Plan:
    """Resolve formulas and hash every sketch and feature, building nothing."""
    r = resolve(design)
    sketches: dict[str, SketchPlan] = {}
    for sk in design.sketches:
        errs = r.error_for(f"sketch:{sk.id}")
        s = sk.model_copy(deep=True)
        needs = False
        vals = r.sketch_values.get(sk.id, {})
        for c in s.constraints:
            if c.expr and c.id in vals:
                if c.value is None or abs(c.value - vals[c.id]) > 1e-9:
                    needs = True
                c.value = vals[c.id]
        if sk.id in r.sketch_offsets:
            s.placement.offset = r.sketch_offsets[sk.id]
        spec = s.model_dump(include={"entities", "constraints", "placement"})
        sketches[sk.id] = SketchPlan(sk.id, _h(spec), s, needs, "; ".join(errs) or None)

    feats: list[FeaturePlan] = []
    prev = _h(rules.model_dump())
    for fi in design.features:
        err = None
        params: dict = {}
        if fi.kind not in FEATURES:
            err = f"unknown feature kind {fi.kind!r}"
        else:
            errs = r.error_for(f"feature:{fi.id}")
            if errs:
                err = "; ".join(errs)
            try:
                given = fi.params | r.feature_values.get(fi.id, {})
                params = resolve_params(fi.kind, given, strict=set(fi.exprs))
            except FeatureError as exc:
                err = err or str(exc)
        refs = {role: sketches[v].hash if v in sketches else v for role, v in fi.refs.items()}
        h = _h({"prev": prev, "kind": fi.kind, "params": params, "refs": refs, "body": fi.body,
                "edges": [e.model_dump(exclude={"points"}) for e in fi.edges], "faces": [f.model_dump() for f in fi.faces],
                "suppressed": fi.suppressed, "error": err})
        feats.append(FeaturePlan(fi.id, h, params, err))
        prev = h
    return Plan(r, sketches, feats)


class Engine:
    """Builds designs, keeping each feature's result (and each solved sketch) for reuse by later builds."""

    def __init__(self, cache_size: int = 256):
        self.cache_size = cache_size
        self._cache: OrderedDict[str, _Snapshot] = OrderedDict()
        self._sketches: OrderedDict[str, Sketch] = OrderedDict()
        self.builds = 0  # Feature.build calls made, for tests and progress
        self.solves = 0  # sketch solves made

    def _remember(self, store: OrderedDict, key: str, value) -> None:
        if self.cache_size <= 0:
            return
        store[key] = value
        store.move_to_end(key)
        while len(store) > self.cache_size:
            store.popitem(last=False)

    def _solved(self, sp: SketchPlan) -> Sketch:
        if not sp.needs_solve:
            return sp.sketch
        hit = self._sketches.get(sp.hash)
        if hit is not None:
            self._sketches.move_to_end(sp.hash)
            return hit
        self.solves += 1
        solved = solve(sp.sketch)
        solved.placement = sp.sketch.placement
        self._remember(self._sketches, sp.hash, solved)
        return solved

    def build(self, design: Design, rules: PrintRules, *, stop_on_fail: bool = True,
              on_feature: Callable[[FeatureResult, Shape | None], None] | None = None, fonts_dir=None) -> BuildResult:
        p = plan(design, rules)
        self._sketch_ids = set(p.sketches)
        sketches: dict[str, Sketch] = {}
        sketch_status: dict[str, str] = {}
        sketch_errors: dict[str, str] = {}
        for sid, sp in p.sketches.items():
            if sp.error:
                sketch_status[sid], sketch_errors[sid] = "fail", sp.error
                continue
            try:
                sketches[sid] = self._solved(sp)
                sketch_status[sid] = "ok"
                if sketches[sid].report is not None and sketches[sid].report.status == "failed":
                    sketch_status[sid], sketch_errors[sid] = "fail", "the sketch's constraints cannot all be met"
            except (SketchSolveError, ValueError) as exc:
                sketch_status[sid], sketch_errors[sid] = "fail", str(exc)

        ctx = BuildContext(rules=rules, sketches=sketches, fonts_dir=fonts_dir, body_ids=tuple(b.id for b in design.bodies))
        results: list[FeatureResult] = []
        checks: list[Check] = []
        failed: str | None = None
        stop = design.rollback if design.rollback is not None else len(design.features)
        for i, (fi, fp) in enumerate(zip(design.features, p.features)):
            res = FeatureResult(id=fi.id, kind=fi.kind, params=fp.params, suppressed=fi.suppressed)
            results.append(res)
            if i >= stop:
                res.status, res.message = "skipped", "after the end-of-part bar"
                continue
            if fi.suppressed:
                res.status, res.message = "skipped", "suppressed"
                continue
            if failed and stop_on_fail:
                res.status, res.message = "skipped", f"{failed} failed before it"
                continue
            hit = self._cache.get(fp.hash)
            if hit is not None:
                self._cache.move_to_end(fp.hash)
                ctx.parts, ctx.outputs, ctx.params = dict(hit.parts), dict(hit.outputs), dict(hit.params)
                res.__dict__.update({k: v for k, v in hit.result.__dict__.items()})
                checks += hit.checks
            else:
                own: list[Check] = []
                self._build_one(ctx, fi, fp, res, own)
                checks += own
                self._remember(self._cache, fp.hash, _Snapshot(dict(ctx.parts), dict(ctx.outputs), dict(ctx.params),
                                                               FeatureResult(**res.__dict__), own))
            if res.status == "fail":
                failed = failed or fi.id
            if on_feature is not None:
                ctx.body = fi.body
                on_feature(res, ctx.part)

        parts = {b: s for b, s in ctx.parts.items() if s is not None}
        hashes = p.hashes()
        if not parts:
            return BuildResult(None, None, results, checks, [], 0.0, 0.0, complete=failed is None, hashes=hashes,
                               param_values=p.resolved.values, errors=p.resolved.errors, sketch_status=sketch_status,
                               sketch_errors=sketch_errors)
        body_names = {b: attribute(s, ctx.outputs) for b, s in parts.items()}
        order = [b.id for b in design.bodies if b.id in parts] + [b for b in parts if not design.body(b)]
        world = parts[order[0]] if len(order) == 1 else Compound(children=[parts[b] for b in order])
        names = [n for b in order for n in body_names[b]]
        volume = sum(parts[b].volume for b in order)
        return BuildResult(world, world, results, checks, names, volume, volume * 1e-9 * rules.density, parts=parts,
                           body_names=body_names, complete=failed is None, hashes=hashes,
                           param_values=p.resolved.values, errors=p.resolved.errors, sketch_status=sketch_status,
                           sketch_errors=sketch_errors)

    def _build_one(self, ctx: BuildContext, fi, fp: FeaturePlan, res: FeatureResult, own: list[Check]) -> None:
        if fp.error:
            res.status, res.error = "fail", fp.error
            return
        sketch_ids = [v for v in fi.refs.values() if v in self._sketch_ids]
        feat = FEATURES[fi.kind]
        ctx.body = fi.body
        ctx.params[fi.id] = fp.params
        ctx.edge_picks = fi.edges
        ctx.face_picks = fi.faces
        self.builds += 1
        try:
            missing = [s for s in sketch_ids if s not in ctx.sketches]
            if missing:
                raise FeatureError(f"sketch {missing[0]} has a problem; fix it first")
            out = feat.build(ctx, fp.params, fi.refs)
            apply_output(ctx, out)
            ctx.outputs[fi.id] = out
            res.driven = out.driven
            own += feat.checks(ctx, fi.id, fp.params, out)
        except FeatureError as exc:
            res.status, res.error = "fail", str(exc)
            return
        except Exception as exc:  # noqa: BLE001 - geometry kernel failures
            res.status, res.error = "fail", f"geometry error: {exc}"
            return
        bad = [c for c in own if c.status != "ok"]
        if bad:
            res.status, res.message = "warn", f"{bad[0].name}: {bad[0].rule}"
