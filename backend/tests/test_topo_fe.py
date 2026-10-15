import numpy as np
import pytest

from print_assist.topo.fe import Grid, Stiffness, hex8_stiffness, solve


def test_hex8_rigid_modes_and_symmetry():
    K = hex8_stiffness(2.0, 70000.0, 0.33)
    assert np.allclose(K, K.T)
    xyz = np.array([(a, b, c) for c in (0, 1) for b in (0, 1) for a in (0, 1)], float) * 2.0
    translate = np.tile([1.0, 0, 0], 8)
    rotate_z = np.column_stack([-xyz[:, 1], xyz[:, 0], 0 * xyz[:, 0]]).ravel()
    assert np.abs(K @ translate).max() < 1e-8 and np.abs(K @ rotate_z).max() < 1e-6
    assert np.sum(np.linalg.eigvalsh(K) < 1e-6) == 6


def _column(nx, ny, nz, h, E):
    """A block on rollers (z = 0 fixed in Z, one corner pinned in X/Y, one in Y) under a top load."""
    g = Grid(nx, ny, nz, h)
    nodes = g.node_coords()
    bottom = np.flatnonzero(nodes[:, 2] == 0)
    fixed = np.concatenate([3 * bottom + 2, [0, 1], [3 * g.node(nx, 0, 0) + 1]])
    st = Stiffness(g, np.ones(g.shape, bool), fixed)
    return g, nodes, st, st.assemble(np.full(len(st.elements), E))


def test_column_under_axial_load_matches_PL_over_AE():
    # 10 x 6 x 40 mm block, E = 70 GPa, 1 kN down, spread to the top nodes as a consistent uniform pressure.
    g, nodes, st, K = _column(5, 3, 20, 2.0, 70000.0)
    top = nodes[:, 2] == g.nz * g.h
    w = np.where(top, 1.0, 0.0)
    on_x_edge = np.isin(nodes[:, 0], (0, g.nx * g.h))
    on_y_edge = np.isin(nodes[:, 1], (0, g.ny * g.h))
    w = w * np.where(on_x_edge, 0.5, 1.0) * np.where(on_y_edge, 0.5, 1.0)  # tributary areas
    f = np.zeros(3 * g.n_nodes)
    f[2::3] = -1000.0 * w / w.sum()
    (u,) = solve(K, st.B, [st.reduce(f)], rtol=1e-10)
    uz = st.expand(u)[2::3][top]
    expected = -1000.0 * 40.0 / (10.0 * 6.0 * 70000.0)  # PL/AE = -0.009524 mm
    assert uz == pytest.approx(np.full(uz.shape, expected), rel=1e-6)


def test_reassembly_scales_with_modulus_and_skips_inactive():
    g = Grid(4, 3, 2, 1.5)
    active = np.ones(g.shape, bool)
    active[3] = False  # last slab of elements removed: its outer nodes vanish from the system
    st = Stiffness(g, active, np.array([], int))
    K1 = st.assemble(np.ones(len(st.elements)))
    K2 = st.assemble(np.full(len(st.elements), 3.0))
    assert np.allclose(K2.toarray(), 3 * K1.toarray())
    assert st.n == 3 * 4 * 4 * 3  # (3+1) x (3+1) x (2+1) nodes left
