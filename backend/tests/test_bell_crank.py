import numpy as np
import pytest
from build123d import IN

from cad_assist.examples import LBF, bell_crank
from cad_assist.project import Project
from cad_assist.topo.fe import solve
from cad_assist.topo.setup import build_setup
from cad_assist.topo.simp import Optimizer


@pytest.fixture(scope="module")
def crank(tmp_path_factory):
    p = Project(tmp_path_factory.mktemp("bell_crank"))
    bell_crank(p)
    return p


def _setup(p, budget=8_000):
    return build_setup(p.data.intent, p.shape, {k: v.role for k, v in p.data.parts.items()}, budget)


def test_bell_crank_example(crank):
    bores = [o for o in crank.data.intent.objects if o.kind == "bore"]
    assert [b.support for b in bores] == ["pin", "free", "link"]
    assert [np.round(np.array(b.frame.origin) / IN, 4).tolist() for b in bores] == [[0, 0, 0], [4, 0, 0], [0, 2.5, 0]]
    assert [round(b.gap / IN, 4) for b in bores] == [0.75, 0.4, 0.4]  # the crank is a male lug in each fork
    (lc,) = [o for o in crank.data.intent.objects if o.kind == "load_case"]
    assert lc.force == pytest.approx((0, -1500 * LBF, 0))


def test_pin_and_link_supports_react_like_a_bell_crank(crank):
    s = _setup(crank)
    assert [(x.id, x.kind, x.along) for x in s.supports] == [("IF-00", "pin", ""), ("IF-02", "link", "X")]
    p = s.problem
    assert len(p.fixed_dofs) == 0 and len(p.springs.extra) == 1
    opt = Optimizer(p)
    st = opt.stiff
    E = np.ones(len(st.elements))  # every non-void element solid
    (u,) = solve(st.assemble(E), st.B, opt.rhs, rtol=1e-10, maxiter=2000, cycle="W")
    u = st.expand(u)
    # The link's pin node is grounded along X (the link) and Z (the pin). Moment balance about the pivot pin, which
    # takes no moment: R · 2.5 in = 1500 lbf · 4.0 in, so the link pushes back with 2400 lbf along X.
    m = s.grid.n_nodes
    sp = p.springs
    ground = (sp.a == m) & (sp.b == -1)
    reaction = -(sp.k[ground, None] * sp.d[ground] * (sp.d[ground] @ u[3 * m:3 * m + 3])[:, None]).sum(axis=0)
    assert reaction[:2] / LBF == pytest.approx([-2400.0, 0.0], abs=0.5)
    # The pin takes the rest: force balance with the 1500 lbf along -Y.
    pin = (sp.b == -1) & (sp.a < m)
    du = u.reshape(-1, 3)[sp.a[pin]]
    r_pin = -(sp.k[pin, None] * sp.d[pin] * np.einsum("ij,ij->i", sp.d[pin], du)[:, None]).sum(axis=0)
    assert r_pin[:2] / LBF == pytest.approx([2400.0, 1500.0], abs=0.5)
    # Out of plane both pins hold Z, so a small self-balancing couple can pass between them.
    assert abs(reaction[2]) < 0.01 * abs(reaction[0]) and r_pin[2] == pytest.approx(-reaction[2], abs=0.01 * LBF)


def test_pin_mounted_design_space(crank):
    from cad_assist.intent.builders import default_design_space

    it = crank.data.intent.model_copy(deep=True)
    it.objects = [o for o in it.objects if o.kind != "design_space"]
    ds = default_design_space(it)
    assert ds.frame.origin == (0, 0, 0) and ds.frame.x == pytest.approx((1, 0, 0))  # at the pivot, square to world
    r, m = 0.377 / 2, 0.5  # bore radius, 0.5 in margin
    assert np.array(ds.min) / IN == pytest.approx([-r - m, -r - m, -0.375 - m], abs=1e-4)
    assert np.array(ds.max) / IN == pytest.approx([4 + r + m, 2.5 + r + m, 0.375 + m], abs=1e-4)


def test_pin_bore_on_a_profile_plate(crank):
    from cad_assist.features.design import build_design
    from cad_assist.features.model import Design
    from cad_assist.features.model import FeatureInstance as FI

    sq = [[-20, -20], [20, -20], [20, 20], [-20, 20]]
    d = Design(features=[FI(id="F1", kind="profile_plate", params={"outline": sq, "z_min": -5.0, "z_max": 5.0, "corner_radius": 0.0}),
                         FI(id="F2", kind="pin_bore", refs={"pin": "IF-00", "base": "F1"})])
    res = build_design(crank.data.intent, d)
    D = 0.377 * IN
    assert res.volume == pytest.approx(40 * 40 * 10 - np.pi * D * D / 4 * 10, rel=1e-6)
    (ed,) = [c for c in res.checks if c.name == "Lug edge e/D"]
    assert ed.value == pytest.approx(20 / D, rel=1e-6)


def test_first_sketches_for_the_bell_crank(crank):
    """The seed drafts sketches that solve cleanly and build a part that clears every fork and keeps full-bearing lugs."""
    from cad_assist.features.design import build_design
    from cad_assist.sketch.solve import solve
    from cad_assist.topo.interpret import Reading
    from cad_assist.topo.setup import MATE_GAP
    from cad_assist.topo.sketch_seed import seed_design

    D = 0.377 * IN
    pins = {"IF-00": (0.0, 0.0, D), "IF-01": (4 * IN, 0.0, D), "IF-02": (0.0, 2.5 * IN, D)}
    z0, z1 = -0.36 * IN, 0.36 * IN
    tine = 0.2 * IN - MATE_GAP  # the input and output forks leave a 0.40 in gap
    ring = [[r * np.cos(a), r * np.sin(a)] for r in (6.0, 12.0) for a in np.linspace(0, 2 * np.pi, 12, endpoint=False)]
    spots = [{"pin": p, "side": sd, "depth": z1 - tine, "radius": 15.0, "columns": ring}
             for p in ("IF-01", "IF-02") for sd in ("+z", "-z")]
    rd = Reading(z0, z1, [], [], [], list(pins), 3.175, pins=pins, web={"+z": 0.25 * IN, "-z": 0.25 * IN}, spots=spots)
    design, notes = seed_design(rd, 1.75)
    design.sketches = [solve(s) for s in design.sketches]
    assert all(s.report.status == "solved" and not s.report.conflicting and not s.report.redundant for s in design.sketches)
    assert [f.kind for f in design.features] == ["pad", "sketch_pocket", "sketch_pocket", "sketch_pocket", "sketch_pocket",
                                                 "pin_bore", "pin_bore", "pin_bore"]
    res = build_design(crank.data.intent, design)
    assert all(f.error is None for f in res.features)
    assert all(c.status == "ok" for c in res.checks if c.name == "Lug edge e/D")
    for name in crank.data.parts:
        assert (res.world & crank.shape(name)).volume < 1e-6, f"part overlaps {name}"
