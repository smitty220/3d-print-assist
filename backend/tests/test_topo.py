from types import SimpleNamespace as NS

import numpy as np
import pytest

from print_assist.intent.models import resolved_support
from print_assist.topo import job as topo
from print_assist.topo.fe import Grid
from print_assist.topo.setup import TopoSetupError, build_setup
from print_assist.topo.simp import Optimizer, Problem
from test_validate import SHAPES, _intent

ROLES = {"rib": "mating", "rod": "mating", "fork": "mating", "panel": "mating", "hyd": "keepout"}


def _setup(it=None, budget=20_000):
    return build_setup(it or _intent(), SHAPES.__getitem__, ROLES, budget)


def test_auto_supports():
    it = _intent()
    mount, bolts, bore = it.objects[0], it.objects[1], it.objects[2]
    assert [resolved_support(it, o) for o in (mount, bolts, bore)] == ["free", "fixed", "free"]
    it.objects[-1].target = bolts.id  # a loaded pattern is not a support unless the user says so
    assert resolved_support(it, bolts) == "free"


def test_element_budget_sets_the_mesh():
    s = _setup(budget=20_000)
    assert 0.85 * 20_000 <= s.active_count <= 20_000
    s2 = _setup(budget=40_000)
    assert s2.grid.h < s.grid.h and s2.active_count <= 40_000
    # The grid never extends past the design space.
    ds = next(o for o in _intent().objects if o.kind == "design_space")
    g = s.grid
    far = s.origin + np.array([g.nx, g.ny, g.nz]) * g.h
    assert np.all(s.origin >= np.array(ds.min) - 1e-9) and np.all(far <= np.array(ds.max) + 1e-9)


def test_interfaces_supports_and_load_resultant():
    it = _intent()
    it.objects[-1].moment = (0.0, 0.0, 5000.0)  # N·mm about the pin centre
    s = _setup(it)
    p, g = s.problem, s.grid
    assert p.solid.sum() > 0 and not (p.solid & p.void).any()
    assert [x.id for x in s.supports] == ["IF-01"] and s.supports[0].kind == "fixed"
    # Each fixed bolt seat is on the mount face (Z = 0).
    z = s.to_world(g.node_coords()[s.supports[0].nodes])[:, 2]
    assert np.allclose(z, 0.0, atol=1e-6)
    (lc,) = s.loads
    assert lc.forces.sum(axis=0) == pytest.approx([0.0, -2800.0, -7740.0], abs=1e-6)
    pin = np.asarray(it.get(lc.target).frame.origin)
    pts = s.to_world(g.node_coords()[lc.nodes])
    assert np.cross(pts - pin, lc.forces).sum(axis=0) == pytest.approx([0.0, 0.0, 5000.0], abs=1e-6)
    # Pin path empty, ear rings solid on both sides of the rod end.
    bore = it.get(lc.target)
    c = s.to_world((np.argwhere(p.solid) + 0.5) * g.h)
    t = (c - pin) @ np.asarray(bore.frame.z)
    near_pin = np.linalg.norm(c - pin - np.outer(t, bore.frame.z), axis=1) < bore.diameter * 1.01
    assert (t[near_pin] > bore.length / 2).any() and (t[near_pin] < -bore.length / 2).any()


def test_setup_refuses_an_incomplete_problem():
    it = _intent()
    it.objects[1].support = "free"
    with pytest.raises(TopoSetupError, match="Nothing holds"):
        _setup(it)
    it = _intent()
    it.objects[-1].force = (0.0, 0.0, 0.0)
    with pytest.raises(TopoSetupError, match="load case"):
        _setup(it)
    it = _intent()
    it.objects = [o for o in it.objects if o.kind != "design_space"]
    with pytest.raises(TopoSetupError, match="design space"):
        _setup(it)


def test_mount_normal_support_fixes_only_z():
    it = _intent()
    it.objects[0].support = "normal"
    s = _setup(it)
    normal = next(x for x in s.supports if x.id == it.objects[0].id)
    assert normal.kind == "normal" and len(normal.nodes) > 100
    assert set(np.unique(s.problem.fixed_dofs % 3)) == {0, 1, 2}  # bolts all three, the face Z only
    assert np.all(np.isin(3 * normal.nodes + 2, s.problem.fixed_dofs))


def test_cantilever_stiffens_at_fixed_volume():
    g = Grid(24, 4, 12, 1.0)
    nodes = g.node_coords()
    fixed_n = np.flatnonzero(nodes[:, 0] == 0)
    f = np.zeros(3 * g.n_nodes)
    tip = np.flatnonzero((nodes[:, 0] == g.nx) & (nodes[:, 2] == 0))
    f[3 * tip + 2] = -1.0 / len(tip)
    p = Problem(g, np.zeros(g.shape, bool), np.zeros(g.shape, bool), (3 * fixed_n[:, None] + np.arange(3)).ravel(), [f])
    r = Optimizer(p).run(0.3, max_iter=25)
    cs = [h.compliance for h in r.history]
    assert cs[0] == pytest.approx(1.0) and cs[-1] < 0.2 * cs[0]  # uniform 30% density carries 0.3³ = 2.7% of solid stiffness
    assert r.history[-1].volume == pytest.approx(0.3, abs=0.005)
    # Material ends up at the root (top and bottom chords), not at mid-height near the free end.
    assert r.x[0, :, -1].mean() > 0.9 and r.x[0, :, 0].mean() > 0.9


def test_runner_sweep_saves_and_meshes(tmp_path, monkeypatch):
    monkeypatch.setattr(topo, "FIRST_ITERS", 6)
    monkeypatch.setattr(topo, "NEXT_ITERS", 4)
    it = _intent()
    proj = NS(root=tmp_path, data=NS(intent=it, parts={k: NS(role=v) for k, v in ROLES.items()}), shape=SHAPES.__getitem__)
    runner = topo.Runner()
    job = runner.start(proj, 5_000, sweep=True, wait=True)
    assert job.status == "done", job.error
    r = runner.result
    assert len(r.steps) == 3 and r.target_step == 1
    assert r.steps[0].fraction > r.steps[1].fraction > r.steps[2].fraction
    assert r.steps[2].mass < r.steps[1].mass < r.steps[0].mass
    assert len(job.history) == 6 + 4 + 4 and job.history[-1]["step"] == 2
    loaded = topo.Result.load(tmp_path / "topo")
    assert loaded.key == r.key and len(loaded.x) == 3 and loaded.target_step == 1
    assert np.abs(loaded.x[1] - r.x[1]).max() <= 0.5 / 255 + 1e-6
    m = loaded.mesh(loaded.x[0], 0.5)
    assert len(m["indices"]) > 300 and m["mass"] > 0
    assert loaded.mesh(loaded.x[0], 0.8)["mass"] < m["mass"]
    # Mesh vertices land inside the design space (Z from the mount face up).
    pts = np.array(m["positions"]).reshape(-1, 3)
    assert pts[:, 2].min() > -1.0
    assert r.key == topo.intent_key(proj)


def test_default_run_is_one_step_at_the_mass_target(tmp_path, monkeypatch):
    monkeypatch.setattr(topo, "FIRST_ITERS", 5)
    it = _intent()
    proj = NS(root=tmp_path, data=NS(intent=it, parts={k: NS(role=v) for k, v in ROLES.items()}), shape=SHAPES.__getitem__)
    runner = topo.Runner()
    job = runner.start(proj, 5_000, wait=True)
    assert job.status == "done" and len(runner.result.steps) == 1 and runner.result.target_step == 0
    # The optimiser holds the volume, so the (grey) mass lands on the target.
    assert runner.result.steps[0].mass == pytest.approx(it.rules.mass_target, rel=0.01)
    it.rules.mass_target = None
    with pytest.raises(topo.TopoRunError, match="mass target"):
        runner.start(proj, 5_000)


def test_run_fractions():
    s = _setup()
    target = 0.36 * 0.45359237
    fr, i = topo.run_fractions(s, target, sweep=False)
    assert i == 0 and fr[0] * s.full_mass() == pytest.approx(target, rel=1e-3)
    fr, i = topo.run_fractions(s, target, sweep=True)
    assert fr == sorted(fr, reverse=True) and len(fr) == 3 and fr[i] * s.full_mass() == pytest.approx(target, rel=1e-3)
    with pytest.raises(topo.TopoRunError, match="between"):
        topo.run_fractions(s, 0.001, sweep=False)


def test_machinable_run_follows_the_raw_one(tmp_path, monkeypatch):
    from print_assist.topo.machining import MachiningSpec
    from print_assist.topo.prismatic import Prismatic25D, setup_faces

    monkeypatch.setattr(topo, "FIRST_ITERS", 12)
    monkeypatch.setattr(topo, "MACHINING_ITERS", 25)
    it = _intent()
    proj = NS(root=tmp_path, data=NS(intent=it, parts={k: NS(role=v) for k, v in ROLES.items()}), shape=SHAPES.__getitem__)
    spec = MachiningSpec(depths=2)
    runner = topo.Runner()
    job = runner.start(proj, 5_000, machining=spec, wait=True)
    assert job.status == "done", job.error
    r = runner.result
    assert [s.label for s in r.steps] == ["raw", "2.5D, 4-axis about Z, 2 floor depths"] and r.target_step == 1
    assert r.machining["depths"] == 2
    assert r.steps[1].mass == pytest.approx(r.steps[0].mass, rel=0.03)  # same mass, so stiffness compares directly
    # As a 0/1 part (interfaces aside) the result is 2.5D: the smallest set of side-face cuts containing it
    # (no blur) adds back almost nothing.
    s = _setup(it, 5_000)
    exact = Prismatic25D(s.problem, setup_faces(2), 2, 1.0)
    design = s.problem.design
    xb = np.where(design, r.x[1] >= 0.5, False).astype(float)
    hull = exact.forward(exact.from_physical(xb))[0] >= 0.5
    added = (hull & ~xb.astype(bool) & design).sum()
    assert added <= 0.02 * xb.sum(), (added, xb.sum())
    with pytest.raises(topo.TopoRunError, match="sweep"):
        runner.start(proj, 5_000, sweep=True, machining=spec)


def test_machining_spec():
    from print_assist.topo.machining import MachiningSpec

    m = MachiningSpec()
    assert m.axis_index((37, 37, 47)) == 2 and m.open_end() == 0
    assert MachiningSpec(axis="x", free_end="+").open_end() == 1
    assert m.filter_radius(1.952) == pytest.approx(6.35 / 2 / 1.952)
    assert MachiningSpec(tool_diameter=2.0).filter_radius(1.952) == 1.5
    assert m.label((37, 37, 47)) == "2.5D, 4-axis about Z, 3 floor depths"
