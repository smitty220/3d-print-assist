import numpy as np
import pytest

from cad_assist.topo.fe import Grid
from cad_assist.topo.prismatic import Prismatic25D, setup_faces
from cad_assist.topo.simp import Problem


def _problem(shape=(6, 5, 8)):
    g = Grid(*shape, 1.0)
    z = np.zeros(shape, bool)
    return Problem(g, z.copy(), z.copy(), np.array([], int), [])


def test_setup_faces():
    assert setup_faces(2) == [(0, 1), (0, -1), (1, 1), (1, -1)]
    assert setup_faces(2, open_end=1)[-1] == (2, 1)


def test_layers_count_from_the_tool_side():
    d = Prismatic25D(_problem(), setup_faces(2), depths=2, filter_radius=1.0)
    plus_x, minus_x = d.faces[0], d.faces[1]
    assert plus_x.layers.tolist() == [1, 1, 1, 0, 0, 0]  # tool at +X: the last three slices are layer 0
    assert minus_x.layers.tolist() == [0, 0, 0, 1, 1, 1]


def test_a_pocket_has_a_flat_floor_and_straight_walls():
    # One face (+Z), two depths, no filter blur: cut layer 0 over a patch, keep layer 1. The result is a
    # rectangular pocket from the top down to the layer boundary, walls straight down.
    d = Prismatic25D(_problem((6, 5, 8)), [(2, 1)], depths=2, filter_radius=1.0)
    (m,) = d.maps(np.ones(d.free.shape))
    m[0, 1:4, 1:3] = 0.0
    x, _ = d.forward(d.flat([m]))
    pocket = x[1:4, 1:3, :]
    assert np.all(pocket[:, :, 4:] == 0) and np.all(pocket[:, :, :4] == 1)
    assert x.sum() == x.size - 3 * 2 * 4
    # Cutting only the deep layer is shadowed by the material above it: nothing changes.
    (m,) = d.maps(np.ones(d.free.shape))
    m[1, 1:4, 1:3] = 0.0
    assert d.forward(d.flat([m]))[0].min() == 1


def test_faces_intersect_and_warm_start_contains_the_shape():
    p = _problem((6, 6, 6))
    d = Prismatic25D(p, setup_faces(2), depths=3, filter_radius=1.0)
    x = np.zeros((6, 6, 6))
    x[2:4, 2:4, :] = 1  # a square post along Z: 2.5D from all four side faces
    z = d.from_physical(x)
    xs, _ = d.forward(z)
    assert np.all(xs[x == 1] >= 1 - 1e-9)
    assert np.allclose(xs[x == 0], 0, atol=1e-5)  # the post is exactly representable


def test_gradient_matches_finite_differences():
    rng = np.random.default_rng(3)
    p = _problem((5, 4, 6))
    p.solid[0, 0, 0] = True
    d = Prismatic25D(p, setup_faces(2, open_end=-1), depths=2, filter_radius=1.6)
    z = rng.uniform(0.1, 0.9, d.free.shape)
    w = rng.normal(size=p.grid.shape)
    x, cache = d.forward(z)
    grad = d.backward(w, cache)
    eps = 1e-6
    for i in rng.choice(z.size, 8, replace=False):
        zp, zn = z.copy(), z.copy()
        zp[i] += eps
        zn[i] -= eps
        fd = ((d.forward(zp)[0] - d.forward(zn)[0]) * w * p.design).sum() / (2 * eps)
        assert grad[i] == pytest.approx(fd, rel=1e-5, abs=1e-9)


def test_a_pale_shallow_layer_does_not_make_the_deeper_one_solid():
    # Regression: an element-by-element shadow ORed one 0.2 map value down a whole slab and came out ~1.
    d = Prismatic25D(_problem((4, 4, 20)), [(2, 1)], depths=2, filter_radius=1.0)
    (m,) = d.maps(np.ones(d.free.shape))
    m[0], m[1] = 0.2, 0.0
    x, _ = d.forward(d.flat([m]))
    assert np.allclose(x, 0.2)
