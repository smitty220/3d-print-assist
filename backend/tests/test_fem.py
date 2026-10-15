"""The stress solver against closed-form answers: a patch test, a cantilever and a plate with a hole."""

import math

import numpy as np
import pytest
from build123d import Box, Cylinder, Pos

from print_assist.fem.element import principal, traction_load, von_mises
from print_assist.fem.mesh import mesh_shape
from print_assist.fem.solve import solve

E, NU = 71_000.0, 0.33  # MPa, 7075-ish


def _face(mesh, axis, value, tol=1e-6):
    """Boundary triangles lying on the plane x[axis] = value."""
    on = np.abs(mesh.nodes[mesh.tris][:, :, axis] - value).max(axis=1) < tol
    return mesh.tris[on]


def _nodes_on(mesh, axis, value, tol=1e-6):
    return np.flatnonzero(np.abs(mesh.nodes[:, axis] - value) < tol)


def test_patch_uniform_tension_is_exact():
    L = 10.0
    mesh = mesh_shape(Pos(L / 2, L / 2, L / 2) * Box(L, L, L), size=4.0)
    s = 50.0
    f = traction_load(mesh.nodes, _face(mesh, 0, L), lambda x: np.tile([s, 0.0, 0.0], (len(x), 1)))
    assert f.reshape(-1, 3).sum(axis=0) == pytest.approx([s * L * L, 0, 0], rel=1e-9)
    # x = 0 face slides in y and z; one corner held, a second held in z: no rigid motion, no extra restraint.
    x0 = _nodes_on(mesh, 0, 0.0)
    corner = x0[np.argmin(np.linalg.norm(mesh.nodes[x0], axis=1))]
    far = x0[np.argmax(mesh.nodes[x0, 1] - mesh.nodes[x0, 2] * 1e-3)]
    fixed = np.unique(np.concatenate([3 * x0, [3 * corner + 1, 3 * corner + 2, 3 * far + 2]]))
    sol = solve(mesh, E, NU, f, fixed)
    assert np.abs(sol.stress[:, 0] - s).max() < 1e-6 * s
    assert np.abs(sol.stress[:, 1:]).max() < 1e-6 * s
    # The far face moves sL/E; the sides shrink by nu times that.
    xL = _nodes_on(mesh, 0, L)
    assert sol.u[3 * xL].mean() == pytest.approx(s * L / E, rel=1e-6)


def test_cantilever_deflection_and_bending_stress():
    L, b, h, P = 200.0, 20.0, 20.0, 1000.0
    mesh = mesh_shape(Pos(L / 2, 0, 0) * Box(L, b, h), size=5.0)
    tip = _face(mesh, 0, L)
    f = traction_load(mesh.nodes, tip, lambda x: np.tile([0.0, 0.0, -P / (b * h)], (len(x), 1)))
    root = _nodes_on(mesh, 0, 0.0)
    fixed = (3 * root[:, None] + np.arange(3)).ravel()
    sol = solve(mesh, E, NU, f, fixed)
    I = b * h**3 / 12
    G = E / (2 * (1 + NU))
    delta = P * L**3 / (3 * E * I) + P * L / (5 / 6 * G * b * h)  # Timoshenko
    w_tip = -sol.u[3 * _nodes_on(mesh, 0, L) + 2].mean()
    assert w_tip == pytest.approx(delta, rel=0.02)
    # Bending stress at mid-span on the top face: M c / I.
    mid = np.flatnonzero((np.abs(mesh.nodes[:, 0] - L / 2) < 6) & (np.abs(mesh.nodes[:, 2] - h / 2) < 1e-6) & (np.abs(mesh.nodes[:, 1]) < b / 4))
    sigma = P * (L / 2) * (h / 2) / I
    assert sol.stress[mid, 0].mean() == pytest.approx(sigma, rel=0.03)
    # Equilibrium: the root reactions balance the load.
    R = sol.reactions.reshape(-1, 3)[root].sum(axis=0)
    assert R == pytest.approx([0, 0, P], abs=1e-6 * P)


def test_plate_with_hole_stress_concentration():
    """Kirsch: a hole in a wide plate under tension peaks at 3x the far-field stress (Peterson, W/D = 8: Kt,gross 3.04)."""
    W, D, t, Lx = 160.0, 20.0, 5.0, 320.0
    from build123d import Mode

    plate = Pos(0, 0, t / 2) * Box(Lx, W, t) - Pos(0, 0, t / 2) * Cylinder(D / 2, t)
    from print_assist.fem.mesh import Refine

    mesh = mesh_shape(plate, size=12.0, curvature=48, refine=[Refine((0, 0, 0), (0, 0, 1), 1.6 * D, 1.5)])
    s = 100.0
    f = traction_load(mesh.nodes, _face(mesh, 0, Lx / 2), lambda x: np.tile([s, 0.0, 0.0], (len(x), 1)))
    f += traction_load(mesh.nodes, _face(mesh, 0, -Lx / 2), lambda x: np.tile([-s, 0.0, 0.0], (len(x), 1)))
    # Self-balanced: hold only rigid-body motion (three points).
    n = mesh.nodes
    a = np.argmin(np.linalg.norm(n - [Lx / 2, 0, 0], axis=1))
    b = np.argmin(np.linalg.norm(n - [-Lx / 2, 0, 0], axis=1))
    c = np.argmin(np.linalg.norm(n - [Lx / 2, W / 2, 0], axis=1))
    fixed = np.array([3 * a + 1, 3 * a + 2, 3 * b + 1, 3 * b + 2, 3 * c + 2, 3 * a])
    sol = solve(mesh, E, NU, f, fixed)
    rim = np.flatnonzero(np.abs(np.linalg.norm(n[:, :2], axis=1) - D / 2) < 1e-3)
    peak = sol.stress[rim, 0].max()
    assert peak / s == pytest.approx(3.04, rel=0.06)
    # The peak is at the hole's edge square to the load, and von Mises / principal agree with sigma_xx there.
    at = rim[np.argmax(sol.stress[rim, 0])]
    assert abs(n[at, 0]) < 2.0
    assert von_mises(sol.stress[[at]])[0] == pytest.approx(peak, rel=0.05)
    assert principal(sol.stress[[at]])[0, 0] == pytest.approx(peak, rel=0.02)


# ---------------------------------------------------------------- the bell crank

from build123d import IN  # noqa: E402

from print_assist.examples import LBF, bell_crank  # noqa: E402
from print_assist.fem.analysis import analyse, bearing_load, bore_tris, pin_forces  # noqa: E402
from print_assist.project import Project  # noqa: E402


@pytest.fixture(scope="module")
def crank(tmp_path_factory):
    """The bell crank example with a simple machined part: tangent outline, one lightening hole, web pockets."""
    from print_assist.features.design import build_design
    from print_assist.sketch.solve import solve as solve_sketch
    from print_assist.topo.recipe import Recipe, compile_recipe

    p = Project(tmp_path_factory.mktemp("crank_fem"))
    bell_crank(p)
    pins = {"IF-00": (0.0, 0.0), "IF-01": (4 * IN, 0.0), "IF-02": (0.0, 2.5 * IN)}
    rec = Recipe.model_validate({
        "z_min": -0.19, "z_max": 0.19, "lugs": {k: 0.57 for k in pins},
        "outline": [{"from": "IF-00", "to": "IF-01"}, {"from": "IF-01", "to": "IF-02"}, {"from": "IF-02", "to": "IF-00"}],
        "holes": [{"points": [[0.9, 0.0], [2.6, 0.0], [0.9, 1.05]], "corner_radius": 0.2}],
    })
    d = compile_recipe(rec, pins)
    d.sketches = [solve_sketch(s) for s in d.sketches]
    p.data.design = d
    p.save()
    b = build_design(p.data.intent, d)
    assert all(f.error is None for f in b.features)
    return p, b


def test_statics_bell_crank(crank):
    p, _b = crank
    (lc,) = [o for o in p.data.intent.objects if o.kind == "load_case"]
    pins = {pf.id: pf for pf in pin_forces(p.data.intent, lc, p.shape)}
    assert pins["IF-01"].role == "load" and pins["IF-00"].role == "pin" and pins["IF-02"].role == "link"
    # Moment about the pivot: link x 2.5 in = 1500 lbf x 4.0 in; the pivot takes the rest.
    assert pins["IF-02"].force / LBF == pytest.approx([-2400, 0, 0], abs=1e-6)
    assert pins["IF-00"].force / LBF == pytest.approx([2400, 1500, 0], abs=1e-6)
    assert sum(pf.force for pf in pins.values()) == pytest.approx([0, 0, 0], abs=1e-6)


def test_bearing_load_resultant_and_side(crank):
    from print_assist.fem.mesh import mesh_shape

    _p, b = crank
    mesh = mesh_shape(b.world, size=5.0)
    r = 0.377 / 2 * IN
    tris = bore_tris(mesh, (4 * IN, 0, 0), (0, 0, 1), r)
    assert len(tris)
    F = np.array([0.0, -1500 * LBF, 0.0])
    f = bearing_load(mesh, tris, (4 * IN, 0, 0), (0, 0, 1), F).reshape(-1, 3)
    assert f.sum(axis=0) == pytest.approx(F, abs=1e-6 * np.linalg.norm(F))
    mag = np.linalg.norm(f, axis=1)
    # The pin pushes the part toward -Y, so it bears on the -Y half of the bore (elements straddling the edge of the
    # loaded half pass a little consistent force to nodes just past it).
    assert mag[mesh.nodes[:, 1] > 0.5].sum() < 0.01 * mag.sum()


def test_analysis_of_the_bell_crank(crank):
    p, b = crank
    mesh, (res,), stats = analyse(p.data.intent, b.world, p.shape, "coarse")
    assert res.residual < 1e-3 * 1500 * LBF  # balanced loads: the three hold points carry (almost) nothing
    names = [r.name for r in res.regions]
    assert {"IF-00 lug", "IF-01 lug", "IF-02 lug"} <= set(names)
    # Hand check on the input lug's net section: P / (2 (R - D/2) t), with a pin-loaded lug's Kt of 2.5-5 on top.
    R, D, t, P = 0.57, 0.377, 0.38, 1500.0
    net = P / (2 * (R - D / 2) * t) * LBF / IN**2  # MPa
    lug = next(r for r in res.regions if r.name == "IF-01 lug")
    assert 2.5 * net < lug.vm < 5.0 * net
    mat = p.data.intent.rules.material_props()
    assert res.ms_ultimate == pytest.approx(mat.ftu / (1.5 * res.vm_max) - 1)


def test_bell_crank_on_pins_in_contact(crank):
    """The bell crank with steel pins in its three forks: the pins' contact forces are the statics reactions, the
    part's slide along the pins is not loaded, and each region's contact / cosine ratio is reported for calibration."""
    p, b = crank
    _mesh, (res,), _stats = analyse(p.data.intent, b.world, p.shape, "coarse", bearing="contact")
    assert res.bearing == "contact" and res.residual < 1e-2
    for pin in res.contact["pins"].values():
        assert pin["force"] == pytest.approx(pin["statics"], abs=1e-3 * 1500 * LBF)
        assert pin["p_max"] > 0 and pin["pin_vm"] > 0
    assert {"IF-00 lug", "IF-01 lug", "IF-02 lug"} <= set(res.calibration)
    assert all(0.7 < k < 2.5 for k in res.calibration.values())


def test_fem_job_and_contour_field(crank):
    import time

    from print_assist.fem.job import FemRunner

    p, _b = crank
    runner = FemRunner()
    job = runner.start(p, "coarse")
    while runner.running():
        time.sleep(0.2)
    assert job.status == "done", job.log
    (c,) = job.cases
    assert c["regions"][0]["vm"] == pytest.approx(c["vm_max"]) and len(c["pins"]) == 3
    f = runner.field(c["id"], "vm")
    n = len(f["positions"]) // 3
    assert len(f["values"]) == n and max(f["indices"]) < n
    # The contour covers the outer surface; the reported peak is over every node, interior ones included.
    assert 0.9 * c["vm_max"] <= f["max"] <= c["vm_max"] * (1 + 1e-9)
    # The viewport flag sits on the surface node with the largest value.
    k = f["peak_index"]
    assert f["values"][k] == pytest.approx(f["max"], abs=1e-4) and f["positions"][3 * k:3 * k + 3] == pytest.approx(f["peak"], abs=1e-3)
    with pytest.raises(KeyError):
        runner.field("LC-99", "vm")
    # A restarted backend picks the last results up from <project>/fem/.
    fresh = FemRunner()
    fresh.attach(p.root / "fem")
    assert fresh.job is not None and fresh.job.cases[0]["vm_max"] == pytest.approx(c["vm_max"])
    g = fresh.field(c["id"], "vm")
    assert len(g["values"]) == len(f["values"]) and g["max"] == pytest.approx(f["max"], rel=1e-5)


def test_sizing_keeps_the_lightest_feasible_part_and_reverts(tmp_path):
    import json
    import time

    from print_assist.fem.job import FemRunner
    from print_assist.sketch.solve import solve as solve_sketch
    from print_assist.topo.recipe import Recipe, compile_recipe

    p = Project(tmp_path / "size")
    bell_crank(p)
    pins = {"IF-00": (0.0, 0.0), "IF-01": (4 * IN, 0.0), "IF-02": (0.0, 2.5 * IN)}
    rec = Recipe.model_validate({
        "z_min": -0.19, "z_max": 0.19, "lugs": {k: 0.6 for k in pins},
        "outline": [{"from": "IF-00", "to": "IF-01"}, {"from": "IF-01", "to": "IF-02"}, {"from": "IF-02", "to": "IF-00"}],
        "holes": [{"points": [[0.9, 0.0], [2.6, 0.0], [0.9, 1.05]], "corner_radius": 0.2}],
        "web": {"rim": 0.2, "depth_top": 0.05, "depth_bottom": 0.05},
    })
    d = compile_recipe(rec, pins)
    d.sketches = [solve_sketch(s) for s in d.sketches]
    d.recipe = json.loads(rec.model_dump_json(by_alias=True))
    p.data.design = d
    p.save()
    m0 = p.build().mass
    runner = FemRunner()
    job = runner.start_sizing(p, target_ms=0.0, max_evals=3)
    while runner.running():
        time.sleep(0.5)
    assert job.status == "done", job.log
    trials = [entry for entry in job.log if entry["kind"] == "trial"]
    assert len(trials) == 3 and job.sizing["feasible"]
    # Lighter than it started (the start already meets MS >= 0 with room to spare), and the kept part is the project's.
    assert job.sizing["after_mass"] < m0 and p.build().mass == pytest.approx(job.sizing["after_mass"])
    assert p.data.design.recipe["web"]["depth_top"] + p.data.design.recipe["web"]["depth_bottom"] > 0.1
    runner.revert_sizing(p)
    assert p.build().mass == pytest.approx(m0)



def test_allowable_overrides():
    from print_assist.intent.models import KSI, Rules

    r = Rules()
    m = r.material_props()
    assert m.ftu == pytest.approx(69 * KSI) and m.fty == pytest.approx(56 * KSI)
    r2 = Rules(ftu=75 * KSI)
    m2 = r2.material_props()
    assert m2.ftu == pytest.approx(75 * KSI) and m2.fty == pytest.approx(56 * KSI) and "overridden" in m2.source


# ---------------------------------------------------------------- mesh controls



def test_edge_divisions_mapped_face_and_sphere():
    from build123d import GeomType

    from print_assist.fem.controls import edge_ref, face_ref, mappable

    plate = Box(60, 30, 4) - Cylinder(5, 4)
    E, F = plate.edges(), plate.faces()
    rims = [i for i, e in enumerate(E) if e.geom_type == GeomType.CIRCLE]
    side = next(i for i, f in enumerate(F) if f.geom_type == GeomType.PLANE and abs(f.normal_at().X) > 0.9)
    top = next(i for i, f in enumerate(F) if f.geom_type == GeomType.PLANE and f.normal_at().Z > 0.9)
    assert mappable(plate, side) is None and "holes" in mappable(plate, top)

    def rim_nodes(m):
        c = m.nodes[np.unique(m.tets[:, :4])]
        return int(((np.abs(np.hypot(c[:, 0], c[:, 1]) - 5) < 1e-3) & (np.abs(c[:, 2] - 2) < 1e-3)).sum())

    base = mesh_shape(plate, size=6.0, curvature=12)
    assert rim_nodes(base) == 12  # curvature alone: 12 per circle
    m = mesh_shape(plate, size=6.0, curvature=12, edges=[dict(n=40, **edge_ref(plate, i)) for i in rims])
    assert rim_nodes(m) == 40  # exactly the divisions asked for
    # Mapped end face (30 x 4 mm): structured, an n1 x n2 grid of cells, two triangles each. Unmapped it is not.
    x = F[side].center().X
    assert len(_face(base, 0, x)) != 2 * 5 * 1
    m = mesh_shape(plate, size=6.0, curvature=12, faces=[face_ref(plate, side)])
    tris = _face(m, 0, x)
    assert len(tris) == 2 * 5 * 1  # ceil(30 / 6) x ceil(4 / 6) cells
    assert len(np.unique(m.nodes[np.unique(tris[:, :3])][:, 1].round(6))) == 6
    # Sphere of influence: elements near its centre are much smaller than the global size.
    m = mesh_shape(plate, size=6.0, curvature=12, spheres=[dict(center=[20, 0, 0], radius=6, size=0.8)])
    edge = lambda t: np.linalg.norm(m.nodes[t[:, 0]] - m.nodes[t[:, 1]], axis=1)  # noqa: E731
    cen = m.nodes[m.tets[:, :4]].mean(axis=1)
    near = np.linalg.norm(cen - [20, 0, 0], axis=1) < 3
    far = np.linalg.norm(cen - [20, 0, 0], axis=1) > 15
    assert edge(m.tets[near]).mean() < 1.5 < 3.0 < edge(m.tets[far]).mean()
    assert m.quality is not None and m.quality.min() > 0
    # A control that matches nothing on the part is skipped and reported.
    m = mesh_shape(plate, size=6.0, curvature=12, edges=[{"ref": [99, 99, 99], "length": 3.0, "n": 5, "label": "ghost"}])
    assert any("ghost" in n for n in m.notes)


def test_rules_proposal_resolve_and_metrics(crank):
    from print_assist.fem.analysis import mesh_metrics, mesh_part
    from print_assist.fem.controls import propose, resolve

    p, b = crank
    s = propose(p.data.intent, b.world, "coarse")
    assert [sp.label for sp in s.spheres] == ["IF-00 lug", "IF-01 lug", "IF-02 lug"]
    assert sum("bore rim" in e.label for e in s.edges) >= 6 and all(e.n >= 3 for e in s.edges)
    found = resolve(s, b.world)
    assert found["edges"] == [e.edge for e in s.edges]  # same part: every control finds its own edge
    # Coarser for the convergence check: same layout, every size x 1.5.
    c = s.scaled(1.5)
    assert c.size == pytest.approx(1.5 * s.size) and c.spheres[0].size == pytest.approx(1.5 * s.spheres[0].size)
    assert c.edges[0].n < s.edges[0].n and c.key() != s.key()
    m = mesh_metrics(mesh_part(b.world, s), p.data.intent)
    bores = {x["id"]: x for x in m["bores"]}
    rim = next(e for e in s.edges if e.label == "IF-01 bore rim")
    around = sum(e.n for e in s.edges if e.label == "IF-01 bore rim") // 2  # two rims, top and bottom
    assert bores["IF-01"]["around"] == around and rim.n > 0
    assert bores["IF-01"]["through"] >= 3  # the best-practice minimum through the lug
    assert m["quality"]["min"] > 0 and sum(m["quality"]["histogram"]) == m["tets"]


def test_solver_reports_convergence(crank):
    from print_assist.fem.analysis import analyse

    p, b = crank
    events = []
    analyse(p.data.intent, b.world, p.shape, "coarse", monitor=lambda stage, **kw: events.append((stage, kw)))
    stages = [s for s, _ in events]
    assert stages.index("mesh") < stages.index("assemble") < stages.index("precondition") < stages.index("iterate") < stages.index("stress")
    res = [kw["residual"] for s, kw in events if s == "iterate"]
    rtol = next(kw["rtol"] for s, kw in events if s == "iterate")
    assert res[0] == 1.0 and res[-1] < rtol < res[-2]  # stops at the first iterate under the tolerance
    assert res[-1] < 1e-3 * res[1]


def test_mesh_job_then_solve_reuses_the_checked_mesh(tmp_path, crank):
    import shutil
    import time

    from print_assist.fem.job import FemRunner

    src, _b = crank
    shutil.copytree(src.root, tmp_path / "p", ignore=shutil.ignore_patterns("fem"))
    p = Project(tmp_path / "p")
    runner = FemRunner()
    job = runner.start_mesh(p)
    while runner.running():
        time.sleep(0.2)
    assert job.status == "done", job.log
    assert p.data.mesh.source == "rules"  # no controls yet: the rules proposed them, saved for the user to see
    view = runner.mesh_payload(p)
    assert not view["stale"] and view["metrics"]["dof"] == job.mesh_stats["dof"]
    n = len(view["positions"]) // 3
    assert max(view["edges"]) < n and len(view["edges"]) % 2 == 0
    job = runner.start(p, "coarse")
    while runner.running():
        time.sleep(0.2)
    assert job.status == "done", job.log
    assert any("Using the checked mesh" in entry["text"] for entry in job.log)
    pr = job.progress
    assert pr["stage"] == "done" and pr["solves"][0]["residual"] < pr["solves"][0]["rtol"]
    assert [s["name"] for s in pr["stages"]][:3] == ["loads", "assemble", "precondition"]
    f = runner.field(job.cases[0]["id"], "vm")
    assert len(f["edges"]) == len(view["edges"])  # results carry the same element edges
    # Edited controls make the checked mesh stale.
    p.data.mesh.size *= 1.2
    assert runner.mesh_payload(p)["stale"]


def test_claude_plans_the_mesh(tmp_path, crank):
    import shutil
    import threading
    from types import SimpleNamespace as NS

    from print_assist.ai.mesh_plan import run_plan
    from print_assist.fem.controls import propose
    from print_assist.fem.job import FemJob, FemRunner

    src, _b = crank
    shutil.copytree(src.root, tmp_path / "p", ignore=shutil.ignore_patterns("fem"))
    p = Project(tmp_path / "p")
    b = p.build()
    rules = propose(p.data.intent, b.world, "coarse")
    flat_with_hole = next(i for i, f in enumerate(b.world.faces()) if len(f.wires()) > 1)

    class Scripted:
        def __init__(self):
            self.calls = []

        def create(self, **kw):
            self.calls.append(kw)
            n = len(self.calls)
            if n == 1:
                return NS(content=[NS(type="text", text="Mapping the top face first."),
                                   NS(type="tool_use", id="t1", name="mesh_and_check",
                                      input={"note": "bad", "setup": {"size": 5, "faces": [{"face": flat_with_hole}]}})])
            if n == 2:
                setup = {"size": 5.0, "curvature": 20, "min_size": 0.25,
                         "edges": [{"edge": f"e{e.edge}", "n": e.n + 4, "label": e.label} for e in rules.edges],
                         "spheres": [s.model_dump() for s in rules.spheres]}
                return NS(content=[NS(type="tool_use", id="t2", name="mesh_and_check", input={"note": "rules plus 4 around", "setup": setup})])
            return NS(content=[NS(type="tool_use", id="t3", name="finish", input={"summary": "Rules with denser bores.", "notes": ["more around the bores"]})])

    client = Scripted()
    job = FemJob(id="x", preset="", convergence=False, mode="plan")
    job.cancel = threading.Event()
    runner = FemRunner()
    s = run_plan(job, p, b, client, "m", "", 3, runner)
    results = {c["tool_use_id"]: c for m in client.calls[-1]["messages"] if m["role"] == "user" and isinstance(m["content"], list)
               for c in m["content"]}
    assert results["t1"]["is_error"] and "cannot be mapped" in results["t1"]["content"]
    assert not results["t2"]["is_error"] and "elements around the bore" in results["t2"]["content"]
    assert s is not None and p.data.mesh.source == "ai" and p.data.mesh.notes == ["more around the bores"]
    assert [e.n for e in p.data.mesh.edges] == [e.n + 4 for e in rules.edges]
    assert runner.mesh_payload(p) is not None and not runner.mesh_payload(p)["stale"]
    assert "Best practices" in client.calls[0]["messages"][0]["content"]


def _bar_joints(F, clearance=None):
    from print_assist.fem.contact import CLEARANCE, Joint

    z = np.array([0, 0, 1.0])
    c = CLEARANCE if clearance is None else clearance
    return z, [Joint("A", "pivot", np.array([10, 0, 0.0]), z, 8.0, 10.0, clearance=c, guess=np.array([-1.0, 0, 0])),
               Joint("B", "load", np.array([70, 0, 0.0]), z, 8.0, 10.0, clearance=c, force=np.array([F, 0, 0]))]


def test_pin_contact_on_a_link():
    """A 6 mm link on two steel pins in forks with a 10 mm gap, pulled with 2 kN: equilibrium through contact, no
    penetration to speak of, a bearing arc under 90 deg that narrows with clearance, and the far field at P / A."""
    from print_assist.fem.analysis import bore_tris
    from print_assist.fem.contact import CLEARANCE, solve_contact

    part = Pos(40, 0, 0) * Box(80, 20, 6) - Pos(10, 0, 0) * Cylinder(4, 10) - Pos(70, 0, 0) * Cylinder(4, 10)
    mesh = mesh_shape(part, size=2.5, curvature=32)
    F = 2000.0
    arcs, crowns = [], []
    for c in (CLEARANCE, 0.010 * 25.4):
        z, joints = _bar_joints(F, c)
        bt = {j.id: bore_tris(mesh, j.origin, z, 4.0) for j in joints}
        r = solve_contact(mesh, joints, bt, E, NU)
        assert r.pin_forces["A"] == pytest.approx([-F, 0, 0], abs=1e-3 * F)
        assert r.pin_forces["B"] == pytest.approx([F, 0, 0], abs=1e-3 * F)
        assert r.hold < 1e-2 and r.penetration < 0.05 * CLEARANCE / 2
        x, p = r.pressure["B"]
        v = x - joints[1].origin
        th = np.degrees(np.abs(np.arctan2(v[:, 1], v[:, 0])))
        mid = np.abs(v[:, 2]) < 1.0
        arcs.append(th[mid & (p > 0)].max())
        crowns.append(p[mid & (th < 10)].mean())
        if c == CLEARANCE:
            mid_bar = (mesh.nodes[:, 0] > 30) & (mesh.nodes[:, 0] < 50)
            assert r.stress[mid_bar, 0].mean() == pytest.approx(F / (20 * 6), rel=0.02)
    # Snug fit: an arc of about +-60 deg and a crown pressure near P / (D t) x 1.1; more clearance, less arc, more peak.
    assert 45 < arcs[0] < 80 and 1.0 < crowns[0] / (F / (8 * 6)) < 1.4
    assert arcs[1] < arcs[0] and crowns[1] > crowns[0]


def test_pin_contact_matches_hertz():
    """A steel pin (R 4) in a 10 mm hole in a 1 mm aluminium slab, 1000 N/mm: non-conforming, so Hertz holds. The thin
    slab is in plane stress: b = sqrt(4 P' R* / (pi E*)), p0 = 2 P' / (pi b), E* = 1 / (1/E1 + 1/E2)."""
    from print_assist.fem.analysis import bore_tris
    from print_assist.fem.contact import PIN_STEEL, Joint, solve_contact

    t, Rb, Rp, Pl, h = 1.0, 5.0, 4.0, 1000.0, 0.12
    mesh = mesh_shape(Box(40, 40, t) - Cylinder(Rb, 2 * t), size=2.0, curvature=48, spheres=[{"center": [Rb, 0, 0], "radius": 1.8, "size": h}])
    z = np.array([0, 0, 1.0])
    j = Joint("P", "load", np.zeros(3), z, 2 * Rb, t, clearance=2 * (Rb - Rp), force=np.array([Pl * t, 0, 0]), stub=0.3,
              spheres=[{"center": [Rp, 0, 0], "radius": 1.8, "size": h}])
    edge = np.flatnonzero(np.abs(mesh.nodes[:, 0] - 20) < 1e-6)
    r = solve_contact(mesh, [j], {"P": bore_tris(mesh, j.origin, z, Rb)}, E, NU, pin_size=1.0,
                      fixed=np.concatenate([3 * edge, 3 * edge + 1, 3 * edge + 2]))
    x, p = r.pressure["P"]
    s = Rb * np.arctan2(x[:, 1], x[:, 0])
    m = p > 0.05 * p.max()
    c1, c0 = np.polyfit(s[m] ** 2, p[m] ** 2, 1)  # p^2 = p0^2 (1 - s^2 / b^2)
    Es = 1 / (1 / PIN_STEEL[0] + 1 / E)
    b = math.sqrt(4 * Pl * (Rp * Rb / (Rb - Rp)) / (math.pi * Es))
    assert math.sqrt(-c0 / c1) == pytest.approx(b, rel=0.03)
    assert math.sqrt(c0) == pytest.approx(2 * Pl / (math.pi * b), rel=0.03)


def test_pin_contact_reacts_the_offset_couple_at_the_forks():
    """The second lug thinned by a top spot face: bearing at different heights leaves a couple the forks react through
    the pins (nothing held but the part's slide along the pins, whose reaction stays ~0)."""
    from print_assist.fem.analysis import bore_tris
    from print_assist.fem.contact import solve_contact

    bar = Pos(40, 0, 0) * Box(80, 20, 6) - Pos(70, 0, 2.5) * Box(20, 20, 1)
    part = bar - Pos(10, 0, 0) * Cylinder(4, 10) - Pos(70, 0, 0) * Cylinder(4, 10)
    mesh = mesh_shape(part, size=2.5, curvature=32)
    F = 2000.0
    z, joints = _bar_joints(F)
    r = solve_contact(mesh, joints, {j.id: bore_tris(mesh, j.origin, z, 4.0) for j in joints}, E, NU)
    assert r.pin_forces["B"] == pytest.approx([F, 0, 0], abs=1e-3 * F) and r.hold < 1e-2
    # The part has no moment support, so the two bearing resultants line up at one height: each lug bears off its own
    # mid-thickness (B's is at -0.5 mm, A's at 0), and the forks take the couple through the pins (pin bending).
    zc = {k: (x[:, 2] * p).sum() / p.sum() for k, (x, p) in r.pressure.items()}
    assert abs(zc["A"] - zc["B"]) < 0.1
    assert zc["B"] > -0.5 + 0.1 and zc["A"] < -0.1


# ---------------------------------------------------------------- converged peaks

def _plate_run(plate, load):
    """`run` for fem.converge: mesh the solid with a setup, apply `load(mesh) -> (f, fixed)`, von Mises per node."""
    from print_assist.fem.analysis import mesh_part

    def run(setup, _round=0):
        m = mesh_part(plate, setup)
        f, fixed = load(m)
        return m, {"LC": von_mises(solve(m, E, NU, f, fixed).stress)}
    return run


def test_converged_peak_at_a_hole_reaches_kirsch():
    """A coarse plate with a hole under-reads the Kirsch peak; refining at the hot spots converges on Kt 3.04."""
    from print_assist.fem.controls import MeshSetup
    from print_assist.fem.converge import converge

    W, D, t, Lx, s = 160.0, 20.0, 5.0, 320.0, 100.0
    plate = Pos(0, 0, t / 2) * Box(Lx, W, t) - Pos(0, 0, t / 2) * Cylinder(D / 2, t)

    def load(m):
        f = traction_load(m.nodes, _face(m, 0, Lx / 2), lambda x: np.tile([s, 0.0, 0.0], (len(x), 1)))
        f += traction_load(m.nodes, _face(m, 0, -Lx / 2), lambda x: np.tile([-s, 0.0, 0.0], (len(x), 1)))
        n = m.nodes
        a, b, c = (np.argmin(np.linalg.norm(n - p, axis=1)) for p in ([Lx / 2, 0, 0], [-Lx / 2, 0, 0], [Lx / 2, W / 2, 0]))
        return f, np.array([3 * a + 1, 3 * a + 2, 3 * b + 1, 3 * b + 2, 3 * c + 2, 3 * a])

    run = _plate_run(plate, load)
    setup = MeshSetup(size=20.0, curvature=6, min_size=0.5)
    mesh, fields = run(setup)
    first = fields["LC"].max()
    log = []
    spots = converge(setup, mesh, fields, run, log=log.append)
    # At the hole's edge square to the load (y = +-D/2; one sphere may take both sides on a mesh this coarse).
    assert 1 <= len(spots) <= 2 and all(abs(abs(sp.center[1]) - D / 2) < 1.0 for sp in spots)
    for sp in spots:
        assert sp.status == "converged", log
        assert sp.history[-1]["h"] < 0.75 * sp.history[0]["h"]  # the mesh really did get finer there
        assert sp.vm / s == pytest.approx(3.04, rel=0.05)
    assert first / s < 2.8  # the coarse mesh alone was well short


def test_converged_peak_flags_a_sharp_inside_corner():
    """An L-shaped cantilever with a sharp inside corner: the peak keeps rising as the mesh refines (stress ~ h^-0.46),
    so the spot is reported as a singularity, not a converged value."""
    from print_assist.fem.controls import MeshSetup
    from print_assist.fem.converge import converge

    t = 5.0
    L = Pos(5, 30, t / 2) * Box(10, 60, t) + Pos(35, 5, t / 2) * Box(50, 10, t)  # leg up the y axis, arm along x

    def load(m):
        f = traction_load(m.nodes, _face(m, 0, 60.0), lambda x: np.tile([0.0, -5.0, 0.0], (len(x), 1)))
        root = _nodes_on(m, 1, 60.0)  # the leg's top end is clamped
        return f, np.concatenate([3 * root, 3 * root + 1, 3 * root + 2])

    run = _plate_run(L, load)
    setup = MeshSetup(size=4.0, curvature=12, min_size=0.2)
    mesh, fields = run(setup)
    log = []
    spots = converge(setup, mesh, fields, run, log=log.append)
    corner = min(spots, key=lambda sp: np.hypot(sp.center[0] - 10, sp.center[1] - 10))
    assert np.hypot(corner.center[0] - 10, corner.center[1] - 10) < 0.5 * corner.history[0]["h"], log
    assert corner.status == "singular", log
    assert corner.extrapolated() is None


def test_fem_job_converges_the_peaks(crank):
    """A coarse bell crank run with peak convergence: every hot spot gets a history of finer meshes and a status, and
    the summary carries the governing converged peak with its margin."""
    import time

    from print_assist.fem.job import FemRunner

    p, _b = crank
    runner = FemRunner()
    job = runner.start(p, "coarse", peaks=True)
    while runner.running():
        time.sleep(0.2)
    assert job.status == "done", job.log
    (c,) = job.cases
    pk = c["peaks"]
    assert pk["bearing"] == "cosine" and not pk["factored"] and pk["spots"]
    for sp in pk["spots"]:
        assert len(sp["history"]) >= 2 and sp["status"] in ("converged", "singular", "limit")
        assert sp["history"][-1]["h"] < sp["history"][0]["h"]
        assert sp["history"][-1]["dof"] > sp["history"][0]["dof"]
    g = pk["governing"]
    mat = p.data.intent.rules.material_props()
    assert g["ms_ultimate"] == pytest.approx(mat.ftu / (1.5 * g["vm"]) - 1)
    # The first mesh's peak is a spot's starting value; the converged peak is within a few percent of it or above.
    top = max(pk["spots"], key=lambda sp: sp["history"][0]["vm"])
    assert top["history"][0]["vm"] == pytest.approx(c["vm_max"], rel=1e-6)
    assert job.progress["peaks"]["spots"]
