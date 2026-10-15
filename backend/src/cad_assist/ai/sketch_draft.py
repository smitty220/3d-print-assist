"""Claude drafts the machined part from a topology result, as a plate recipe (topo/recipe.py), and asks for feedback.

Claude gets measurements, not pictures: pins, supports and the load, the thickness floors and what each fork needs
cleared, the result's cut-outs, the rules, a text map of the result's thickness, and the rules-based draft as a
baseline recipe. Each build_and_check compiles the recipe into the constrained sketches and features the engineer
edits, builds the part, validates it and compares it with the result on the topology mesh (mass, stiffness, and a map
of where they differ). Candidates appear in the viewport as they are built; the best one is kept.
"""

from __future__ import annotations

import json
import math

import numpy as np
from pydantic import ValidationError
from skimage.measure import approximate_polygon

from ..features.context import FeatureError
from ..features.design import build_design
from ..sketch.solve import solve
from ..topo.interpret import IN, LB, Reading, _inch, check_model
from ..topo.recipe import Hole, OutlineEdge, Recipe, RecipeError, Spot, Web, compile_recipe
from ..validate import validate

MAX_BUILDS = 5

SYSTEM = """You are the design engine inside CAD Assist, turning a topology-optimisation result into a machined part.

The engineer is a career aerospace stress engineer who works with machine shops. They ran a 2.5D topology \
optimisation of a pin-mounted plate part (a bell crank or lever). Your job: draft the part a good designer would \
draw from that result, as a plate recipe that the tool compiles into constrained sketches (lines, arcs, splines, \
dimensions) and pad / pocket features. The engineer then edits those sketches by hand, so favour geometry they would \
recognise and want to keep: few, meaningful edges; standard radii; straight lines where the load path is straight; \
arcs or a spline where it curves; flat-floored pockets; cutter-radius corners.

Units: inches throughout. The sketch plane is the design frame's XY; pins run along Z.

The recipe (build_and_check's `recipe`):
- z_min, z_max: stock thickness range. Keep the hub between its fork's tines (see the measured stock range).
- lugs: a radius per pin. Edge distance e = lug radius; e/D >= 1.5 is full bearing (the rule), 1.0 the minimum.
- outline: one edge per pair of neighbouring pins, counter-clockwise round the part, in the order given. edge is \
"tangent" (straight, tangent to both lugs), "arc" with radius and bulge "out" (convex, bowing outward) or "in" \
(concave), or "spline" with through points (it leaves and arrives tangent to the lugs). Follow where the result has \
material: an outer flange that bows outward wants a convex arc or a spline; a side where the result carries nothing \
can come in (concave) to save mass.
- holes: lightening holes, each a list of corner points (a triangle or quadrilateral is usual) and a corner_radius \
(at least the cutter radius). Put them where the result has none; leave at least the rim width from the outline.
- web: rim (wall left inside the outline), boss_gap (extra clearance round each lug's boss), depth_top and \
depth_bottom (pocket depths from +Z and -Z). The web left is z_max - z_min - depth_top - depth_bottom.
- spots: spot faces thinning a lug to fit its fork: pin, side (+z / -z), depth, and step (distance from the pin to a \
step square to the arm). Measured needs are given; don't go shallower, or the part hits the fork.
- corner_radius: the cutter radius.

Each build reports: build errors; mass against the topology result's mass; stiffness against the topology result \
(same supports and load, both as voxels; 1.00 = as stiff); every validator failure and warning; and a map of where \
your part has material the result did not ('+') and is missing material the result had ('-').

When stress checking is on, each build also reports a coarse stress analysis (quadratic tetrahedra, pin forces as bearing pressure, reactions from statics): peak von Mises by region and the margins of safety (limit vs yield, ultimate = factor x limit vs ultimate strength). Keep MS ultimate at or above the target given; where a region has a large margin, take material out of it; where it is short, add it there (a bigger lug, a thicker web, a wider rim).

Working method: start from the baseline recipe (the rules-based draft, which passes the checks). Make deliberate \
changes, a few per build, saying why in one short sentence. Priorities: no failures; stiffness at least about the \
topology result's; then mass near the topology result's; then simplicity and machinability. Use at most the builds \
allowed, then call finish with a short summary and the questions you want the engineer to decide (two or three, \
concrete: "Keep the left side concave (saves 0.02 lb) or straight?"). Keep notes short: the engineer reads them live."""

TOOLS = [
    {
        "name": "build_and_check",
        "description": "Compile a plate recipe into sketches and features, build the part, validate it and compare it with the topology result.",
        "input_schema": {
            "type": "object",
            "properties": {
                "note": {"type": "string", "description": "One sentence for the engineer: what this build changes and why."},
                "recipe": {"type": "object", "description": "The full plate recipe (see the system prompt)."},
            },
            "required": ["note", "recipe"],
        },
    },
    {
        "name": "finish",
        "description": "End the session; the best part built is kept in the project.",
        "input_schema": {
            "type": "object",
            "properties": {
                "summary": {"type": "string", "description": "2-4 sentences: the part, its mass and stiffness against the topology result, anything left open."},
                "questions": {"type": "array", "items": {"type": "string"}, "description": "Decisions for the engineer (two or three)."},
            },
            "required": ["summary"],
        },
    },
]


def pin_order(pins: dict[str, tuple[float, float, float]]) -> list[str]:
    c = np.mean([(x, y) for x, y, _d in pins.values()], axis=0)
    return sorted(pins, key=lambda k: math.atan2(pins[k][1] - c[1], pins[k][0] - c[0]))


def baseline_recipe(rd: Reading, h: float) -> Recipe:
    """The rules-based draft (as topo/sketch_seed.py drafts it) written as a recipe: Claude's starting point."""
    order = pin_order(rd.pins)
    r_lug = math.ceil(max(1.5 * d for _x, _y, d in rd.pins.values()) / (0.005 * IN)) * 0.005  # in
    z_min, z_max = rd.z_min, rd.z_max
    spots = list(rd.spots)
    for side in ("+z", "-z"):
        mine = [s for s in spots if s["side"] == side]
        if len({s["pin"] for s in mine}) == len(rd.pins):
            skim = min(s["depth"] for s in mine)
            z_max, z_min = (z_max - skim, z_min) if side == "+z" else (z_max, z_min + skim)
            spots = [dict(s, depth=s["depth"] - skim) if s["side"] == side else s for s in spots]
    spots = [s for s in spots if s["depth"] > 1e-3]
    web_depth = {sd: d - ((rd.z_max - z_max) if sd == "+z" else (z_min - rd.z_min)) for sd, d in rd.web.items()}
    centroid = np.mean([(x, y) for x, y, _d in rd.pins.values()], axis=0)
    spot_list = []
    for s in spots:
        px, py, _d = rd.pins[s["pin"]]
        u = np.array([px, py]) - centroid
        u /= np.linalg.norm(u)
        t0 = float(min(np.asarray(s["columns"]) @ u)) - h / 2 - rd.corner_radius
        spot_list.append(Spot(pin=s["pin"], side=s["side"], depth=round(s["depth"] / IN, 4), step=round(-t0 / IN, 4)))
    holes = []
    for c in rd.cutouts:
        pts = approximate_polygon(np.asarray(c), tolerance=4.0 * h)[:-1]
        if len(pts) >= 3:
            holes.append(Hole(points=[(round(x / IN, 3), round(y / IN, 3)) for x, y in pts], corner_radius=round(rd.corner_radius / IN, 4)))
    return Recipe(
        z_min=round(z_min / IN, 4), z_max=round(z_max / IN, 4),
        lugs={k: r_lug for k in rd.pins},
        outline=[OutlineEdge(**{"from": a, "to": b, "edge": "tangent"}) for a, b in zip(order, order[1:] + order[:1])],
        holes=holes,
        web=Web(rim=0.15, boss_gap=0.0, depth_top=round(max(0.0, web_depth.get("+z", 0)) / IN, 4),
                depth_bottom=round(max(0.0, web_depth.get("-z", 0)) / IN, 4)),
        spots=spot_list,
        corner_radius=round(rd.corner_radius / IN, 4),
    )


def _grid_map(x: np.ndarray, result, step: int = 2, model: np.ndarray | None = None, pins=None) -> str:
    """Text map of the result (thickness 1-9, '.' empty), or of model vs result ('#' both, '+' model only, '-' result
    only), rows from high Y to low, `step` elements per character, pins as 'O'."""
    th = x.sum(axis=2)
    nz = x.shape[2]
    mm = model.any(axis=2) if model is not None else None
    nx, ny = th.shape
    rows = []
    o = np.asarray(result.origin)
    for j in range(ny - 1, -1, -step):
        row = ""
        for i in range(0, nx, step):
            xc, yc = o[0] + (i + 0.5) * result.h, o[1] + (j + 0.5) * result.h
            if pins and any(math.hypot(xc - px, yc - py) < d / 2 + result.h for px, py, d in pins.values()):
                row += "O"
                continue
            t = th[i:i + step, max(0, j - step + 1):j + 1].max()
            if mm is None:
                row += "." if t == 0 else str(min(9, max(1, round(9 * t / nz))))
            else:
                m = mm[i:i + step, max(0, j - step + 1):j + 1].any()
                row += "#" if (t and m) else "+" if m else "-" if t else "."
        rows.append(row)
    return "\n".join(rows)


def brief(rd: Reading, result, step: int, intent, base: Recipe, current: dict | None, edited: bool, instruction: str, max_builds: int,
          last_summary: str = "", last_questions: list[str] | None = None) -> str:
    x = result.x[step] >= 0.5
    ref = result.steps[step]
    sup = {o.id: getattr(o, "support", "") for o in intent.objects if o.kind == "bore"}
    loads = [f"{lc.id} at {lc.target}: ({', '.join(f'{v / 4.4482216:.0f}' for v in lc.force)}) lbf" for lc in intent.objects if lc.kind == "load_case"]
    order = pin_order(rd.pins)
    ds = next((o for o in intent.objects if o.kind == "design_space"), None)
    lines = []
    if last_questions or last_summary:
        lines += ["Your last draft's summary: " + (last_summary or "(none)"), "The questions you asked the engineer:"]
        lines += [f"{i}. {q}" for i, q in enumerate(last_questions or [], 1)]
        lines += ["The engineer's reply follows (they may answer by number).", ""]
    lines += [
        f"Engineer's request: {instruction.strip() or 'Draft the part from the topology result.'}",
        "",
        "Pins (x, y in; diameter; support): " + "; ".join(
            f"{k} ({rd.pins[k][0] / IN:.3f}, {rd.pins[k][1] / IN:.3f}) Ø{rd.pins[k][2] / IN:.3f} {sup.get(k) or 'free'}" for k in order),
        "Loads: " + "; ".join(loads),
        "Outline edge order (counter-clockwise): " + ", ".join(f"{a}→{b}" for a, b in zip(order, order[1:] + order[:1])),
        f"Measured stock (between the forks' tines): Z {rd.z_min / IN:.3f} to {rd.z_max / IN:.3f} in",
        "Spot faces the forks need (depth from the measured stock face; columns from the pin): " + "; ".join(
            f"{s['pin']} {s['side']} {s['depth'] / IN:.3f} in" for s in rd.spots),
        "Floors covering most of each face in the result: " + ", ".join(f"{d / IN:.3f} in from {sd}" for sd, d in rd.web.items()),
        "Result cut-outs (corner points, in): " + json.dumps([[[round(a / IN, 2), round(b / IN, 2)] for a, b in approximate_polygon(np.asarray(c), 2 * result.h)[:-1]] for c in rd.cutouts]),
        f"Topology result: {ref.label}, {ref.solid_mass / LB:.3f} lb as solid voxels (the mass to aim near)",
        f"Rules: cutter radius {rd.corner_radius / IN:.3f} in; lug e/D ≥ {intent.rules.lug_edge_full_bearing:g} full bearing; "
        f"min wall {intent.rules.min_wall_preferred / IN:.3f} in preferred, {intent.rules.min_wall / IN:.3f} min",
    ]
    if ds is not None:
        lines.append(f"Design space (the part must stay inside): X {ds.min[0] / IN:.2f} to {ds.max[0] / IN:.2f}, Y {ds.min[1] / IN:.2f} to {ds.max[1] / IN:.2f} in")
    lines += [
        f"Builds allowed: {max_builds}",
        "",
        f"Result thickness map (1-9 = thin to full stock, '.' empty, 'O' pins; {2 * result.h / IN:.3f} in per character; "
        f"top row y = {(result.origin[1] + result.grid[1] * result.h) / IN:.2f} in, left column x = {result.origin[0] / IN:.2f} in):",
        _grid_map(x, result, pins=rd.pins),
        "",
        "Baseline recipe (rules-based draft, passes the checks):",
        base.model_dump_json(by_alias=True),
    ]
    if current is not None:
        lines += ["", "The recipe of the current design (your last draft):", json.dumps(current)]
        if edited:
            lines.append("The engineer has edited the sketches by hand since that draft; a new build replaces their edits, so keep "
                         "what they asked for and mention the edits in your notes.")
    return "\n".join(lines)


def _score(fails: int, ratio: float, mass: float, ref_mass: float, warns: int) -> tuple:
    return (fails, max(0.0, 0.97 - ratio), round(max(0.0, mass - ref_mass) / LB, 3), warns, mass)


def run_draft(job, project, result, step: int, rd: Reading, setup_of, instruction: str, client, max_builds: int = MAX_BUILDS,
              model: str = "", on_build=None, stress: bool = False, target_ms: float = 0.10) -> dict | None:
    """The session: Claude builds recipes until it calls finish (or runs out of builds). Logs into `job` (an
    InterpretJob). Returns {design, recipe, check, build} for the best part, or None."""
    from ..sketch.model import Sketch  # noqa: F401 - keeps the import graph obvious

    intent = project.data.intent
    roles = {k: v.role for k, v in project.data.parts.items()}
    pins = {k: (x, y) for k, (x, y, _d) in rd.pins.items()}
    base = baseline_recipe(rd, result.h)
    cur = project.data.design.recipe
    edited = bool(cur) and project.data.design.recipe_key != project.data.design.sketch_key()
    ref = result.steps[step]
    setup = setup_of()
    x = result.x[step] >= 0.5
    best: dict | None = None
    builds = 0

    system = [{"type": "text", "text": SYSTEM, "cache_control": {"type": "ephemeral"}}]
    d0 = project.data.design
    last = (d0.recipe_summary, d0.recipe_questions) if cur else ("", [])
    first = brief(rd, result, step, intent, base, cur, edited, instruction, max_builds, *last)
    if stress:
        first += f"\n\nStress checking is on: target MS ultimate {target_ms:+.2f}."
    messages = [{"role": "user", "content": first}]
    if edited:
        job.say("skip", "You have edited the sketches since Claude's last draft; a new draft replaces those edits")

    def build(args) -> tuple[str, bool]:
        nonlocal best, builds
        if builds >= max_builds:
            return "Build limit reached. Call finish now.", True
        try:
            rec = Recipe.model_validate(args.get("recipe") or {})
        except ValidationError as exc:
            return f"Invalid recipe: {exc}", True
        builds += 1
        note = (args.get("note") or "").strip()
        try:
            design = compile_recipe(rec, pins)
            design.sketches = [solve(s) for s in design.sketches]
            bad = [f"{s.id} {s.name}: {s.report.status}, conflicting {s.report.conflicting}" for s in design.sketches
                   if s.report.status != "solved" or s.report.conflicting]
            b = build_design(intent, design)
        except (RecipeError, FeatureError, ValueError) as exc:
            job.say("skip", f"Build {builds}: {exc}")
            return f"Build {builds} failed: {exc}", True
        errs = [f"{f.id} {f.kind}: {f.error}" for f in b.features if f.error]
        if errs or b.world is None:
            job.say("skip", f"Build {builds}: {'; '.join(errs) or 'nothing built'}")
            return f"Build {builds} failed: {'; '.join(errs) or 'nothing built'}", True
        report = validate(intent, b, project.shape, roles)
        mc = check_model(setup, result, step, b.world)
        fails = [f for f in report.part + report.preflight if f.status == "fail"]
        warns = [f for f in report.part if f.status == "warn"]
        design.recipe = json.loads(rec.model_dump_json(by_alias=True))
        design.recipe_key = design.sketch_key()
        stress_lines, short = [], 0
        if stress:
            from ..fem.analysis import AnalysisError, analyse

            try:
                _m, (sr, *_), _st = analyse(intent, b.world, project.shape, "coarse")
                short = int(sr.ms_ultimate < target_ms) + int(sr.ms_yield < 0)
                stress_lines = [f"Stress (coarse FEA, {sr.category} load): MS ultimate {sr.ms_ultimate:+.2f}, MS yield {sr.ms_yield:+.2f}; "
                                f"peak von Mises {sr.vm_max / 6.894757:.1f} ksi"]
                stress_lines += [f"  {g.name}: {g.vm / 6.894757:.1f} ksi von Mises" for g in sr.regions]
                note += f" · MS ult {sr.ms_ultimate:+.2f} ({sr.regions[0].name if sr.regions else '?'})"
            except AnalysisError as exc:
                stress_lines = [f"Stress analysis not possible: {exc}"]
        score = _score(len(fails) + short, mc.stiffness_ratio, b.mass, ref.solid_mass, len(warns))
        if best is None or score < best["score"]:
            best = {"score": score, "design": design, "recipe": design.recipe, "check": mc, "build": b, "n": builds}
        if on_build is not None:
            on_build(design, b)
        job.say("build", f"Build {builds}: {note} · {b.mass / LB:.3f} lb, stiffness {mc.stiffness_ratio - 1:+.1%} vs result, "
                         f"{len(fails)} failing, {len(warns)} warnings")
        lines = [f"Build {builds} of {max_builds}.",
                 f"Mass {b.mass / LB:.3f} lb exact ({mc.mass / LB:.3f} lb as voxels) vs result {ref.solid_mass / LB:.3f} lb",
                 f"Stiffness vs result: {mc.stiffness_ratio:.3f}"]
        lines += [f"[fail] {f.title}: {f.detail}" for f in fails] + [f"[warn] {f.title}: {f.detail}" for f in warns]
        lines += [f"Sketch problem: {m}" for m in bad]
        lines += stress_lines
        lines += ["Model vs result ('#' both, '+' model only, '-' result only, 'O' pins):",
                  _grid_map(x, result, model=mc.solid, pins=rd.pins)]
        return "\n".join(lines), False

    tokens = {"input": 0, "output": 0, "cache_read": 0}
    for _turn in range(max_builds * 3 + 4):
        if job.cancel.is_set():
            job.say("skip", "Stopped")
            break
        resp = client.create(model=model, max_tokens=8000, system=system, tools=TOOLS, messages=messages)
        u = getattr(resp, "usage", None)
        if u is not None:
            tokens["input"] += getattr(u, "input_tokens", 0) or 0
            tokens["output"] += getattr(u, "output_tokens", 0) or 0
            tokens["cache_read"] += getattr(u, "cache_read_input_tokens", 0) or 0
            job.tokens = dict(tokens)
        messages.append({"role": "assistant", "content": [_block(b) for b in resp.content]})
        results, done = [], False
        for blk in resp.content:
            if blk.type == "text" and blk.text.strip():
                job.say("ai", blk.text.strip())
            elif blk.type == "tool_use":
                if blk.name == "build_and_check":
                    text, err = build(blk.input or {})
                elif blk.name == "finish":
                    args = blk.input or {}
                    summary = (args.get("summary") or "").strip()
                    questions = [str(q).strip() for q in args.get("questions") or [] if str(q).strip()]
                    job.say("done", summary)
                    for i, q in enumerate(questions, 1):
                        job.say("question", f"{i}. {q}")
                    if best is not None:
                        best["design"].recipe_summary, best["design"].recipe_questions = summary, questions
                    text, err, done = "Session ended.", False, True
                else:
                    text, err = f"Unknown tool {blk.name!r}", True
                results.append({"type": "tool_result", "tool_use_id": blk.id, "content": text, "is_error": err})
        if done or not results:
            break
        messages.append({"role": "user", "content": results})
    return best


def _block(b) -> dict:
    if b.type == "text":
        return {"type": "text", "text": b.text}
    if b.type == "tool_use":
        return {"type": "tool_use", "id": b.id, "name": b.name, "input": b.input}
    return b.model_dump() if hasattr(b, "model_dump") else dict(b)
