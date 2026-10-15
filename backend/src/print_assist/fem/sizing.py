"""Sizing: the lightest version of a drafted part that still meets a margin of safety.

Works on the plate recipe the part was drafted from (topo/recipe.py). Variables: web thickness, rim width and each
lug's radius (never below the full-bearing edge distance rule, but grown when a lug governs). Every trial recompiles
the recipe into sketches and features, rebuilds, validates and runs a coarse stress analysis. Feasible = MS ultimate
>= target, MS yield >= 0, no validator failures, web and rim at least the preferred minimum wall. Method: grow what
governs until feasible, then coordinate search toward less mass, halving a variable's step when a move fails.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field

from ..features.design import build_design
from ..sketch.solve import solve as solve_sketch
from ..topo.recipe import Recipe, RecipeError, compile_recipe
from ..validate import validate
from .analysis import AnalysisError, analyse

IN = 25.4
LB = 0.45359237
STEPS = {"web": 0.04, "rim": 0.04, "lug": 0.05}  # in
MIN_STEP = 0.01  # in


@dataclass
class Trial:
    vars: dict
    mass: float  # kg
    ms_ult: float
    ms_yield: float
    fails: int
    governing: str  # region with the lowest ultimate margin
    feasible: bool
    design: object = field(repr=False, default=None)
    build: object = field(repr=False, default=None)


def _vars(rec: Recipe) -> dict:
    web = rec.web
    t = rec.z_max - rec.z_min
    v = {"web": round(t - (web.depth_top + web.depth_bottom), 4) if web else round(t, 4), "rim": web.rim if web else 0.15}
    for k, r in rec.lugs.items():
        v[f"lug {k}"] = r
    return v


def _apply(rec: Recipe, v: dict) -> Recipe:
    r = rec.model_copy(deep=True)
    t = r.z_max - r.z_min
    if r.web is not None:
        cut = max(0.0, t - v["web"])
        old = r.web.depth_top + r.web.depth_bottom
        share = r.web.depth_top / old if old > 0 else 0.5
        r.web.depth_top, r.web.depth_bottom = round(cut * share, 4), round(cut * (1 - share), 4)
        r.web.rim = round(v["rim"], 4)
    for k in r.lugs:
        r.lugs[k] = round(v[f"lug {k}"], 4)
    return r


def size_part(project, pins: dict, target_ms: float = 0.10, max_evals: int = 20, log=None, cancel=None) -> tuple[Trial | None, list[Trial]]:
    """Returns the best feasible trial (or the least infeasible one) and every trial."""
    say = log or (lambda *_: None)
    d0 = project.data.design
    if not d0.recipe:
        raise AnalysisError("Sizing works on a drafted part (Draft with AI); this design has no recipe")
    rec0 = Recipe.model_validate(d0.recipe)
    intent = project.data.intent
    rules = intent.rules
    roles = {k: v.role for k, v in project.data.parts.items()}
    bores = {o.id: o for o in intent.objects if o.kind == "bore"}
    wall = rules.min_wall_preferred / IN
    lug_min = {k: rules.lug_edge_full_bearing * bores[k].diameter / IN for k in rec0.lugs if k in bores}
    trials: list[Trial] = []

    def evaluate(v: dict) -> Trial:
        rec = _apply(rec0, v)
        try:
            design = compile_recipe(rec, pins)
            design.sketches = [solve_sketch(s) for s in design.sketches]
            b = build_design(intent, design)
            if any(f.error for f in b.features) or b.world is None:
                raise RecipeError("; ".join(f"{f.id}: {f.error}" for f in b.features if f.error) or "nothing built")
            report = validate(intent, b, project.shape, roles)
            fails = sum(f.status == "fail" for f in report.part + report.preflight)
            _m, (res, *_), _st = analyse(intent, b.world, project.shape, "coarse")
        except (RecipeError, AnalysisError, ValueError) as exc:
            say("trial", f"{_fmt(v)}: does not build ({exc})")
            t = Trial(v, float("inf"), -1.0, -1.0, 99, "build", False)
            trials.append(t)
            return t
        design.recipe = json.loads(rec.model_dump_json(by_alias=True))
        design.recipe_key = design.sketch_key()
        design.recipe_summary, design.recipe_questions = d0.recipe_summary, d0.recipe_questions
        regions = sorted(res.regions, key=lambda g: g.vm, reverse=True)
        gov = regions[0].name if regions else "?"
        ok = res.ms_ultimate >= target_ms and res.ms_yield >= 0 and fails == 0 and v["web"] >= wall - 1e-9 and v["rim"] >= wall - 1e-9
        t = Trial(dict(v), b.mass, res.ms_ultimate, res.ms_yield, fails, gov, ok, design, b)
        trials.append(t)
        say("trial", f"{_fmt(v)}: {b.mass / LB:.3f} lb, MS ult {res.ms_ultimate:+.2f} ({gov}), {fails} failing"
                     f"{' ✓' if ok else ''}")
        return t

    def better(a: Trial, b: Trial | None) -> bool:
        if b is None:
            return True
        if a.feasible != b.feasible:
            return a.feasible
        if a.feasible:
            return a.mass < b.mass
        return (a.fails, -a.ms_ult) < (b.fails, -b.ms_ult)

    v = _vars(rec0)
    say("read", f"Start: {_fmt(v)}; target MS ultimate {target_ms:+.2f}, at most {max_evals} trials (a coarse analysis each)")
    cur = evaluate(v)
    best = cur
    # Grow what governs until feasible.
    while not cur.feasible and len(trials) < max_evals and not (cancel and cancel.is_set()):
        nv = dict(cur.vars)
        if cur.governing.endswith(" lug") and f"lug {cur.governing[:-4]}" in nv:
            nv[f"lug {cur.governing[:-4]}"] += STEPS["lug"]
        elif cur.fails and cur.ms_ult >= target_ms:
            nv["rim"] += STEPS["rim"]  # failures with margin to spare: usually a thin wall
        else:
            nv["web"] += STEPS["web"]
        cur = evaluate(nv)
        if better(cur, best):
            best = cur
    if not cur.feasible:
        say("result", "No feasible design within the trial budget; kept the closest")
        return best, trials
    # Coordinate search toward less mass.
    steps = {k: STEPS["lug" if k.startswith("lug") else k] for k in cur.vars}
    lower = {"web": wall, "rim": wall} | {f"lug {k}": m for k, m in lug_min.items()}
    while len(trials) < max_evals and any(s >= MIN_STEP for s in steps.values()) and not (cancel and cancel.is_set()):
        moved = False
        for k in list(steps):
            if steps[k] < MIN_STEP or len(trials) >= max_evals:
                continue
            nv = dict(best.vars)
            nv[k] = max(lower.get(k, 0.0), nv[k] - steps[k])
            if best.vars[k] - nv[k] < MIN_STEP:  # at its lower limit: not worth a trial
                steps[k] = 0.0
                continue
            t = evaluate(nv)
            if t.feasible and t.mass < best.mass:
                best, moved = t, True
            else:
                steps[k] /= 2
        if not moved and all(s < MIN_STEP for s in steps.values()):
            break
    return best, trials


def _fmt(v: dict) -> str:
    parts = [f"web {v['web']:.3f}", f"rim {v['rim']:.3f}"]
    parts += [f"{k.split(' ', 1)[1]} R{v[k]:.3f}" for k in v if k.startswith("lug ")]
    return ", ".join(parts) + " in"
