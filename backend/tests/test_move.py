"""Move / rotate: a body turned about any axis (point + direction, or a picked edge), then moved; or a moved copy into
another body. Centres and volumes are hand-checked."""

import pytest

from print_assist.build.ops import apply
from print_assist.features.design import build_design
from print_assist.features.model import Design


def block(d: Design) -> None:
    """A 20 x 20 x 40 block centred on the origin, standing on the bed: centre (0, 0, 20), 16000 mm3."""
    apply(d, {"op": "add_feature", "kind": "plate", "params": {"length": 20, "width": 20, "thickness": 40, "corner_radius": 0}})


def built(d: Design):
    r = build_design(d)
    assert r.complete, [f.error for f in r.features]
    return r


def centre(shape) -> tuple[float, ...]:
    c = shape.center()
    return tuple(round(v, 6) + 0.0 for v in (c.X, c.Y, c.Z))


def test_turn_about_a_vertical_axis_through_a_point():
    d = Design()
    block(d)
    apply(d, {"op": "add_feature", "kind": "move", "params": {"angle": 90, "axis_x": 10}})
    r = built(d)
    # (0, 0) turned +90 about (10, 0): the offset (-10, 0) becomes (0, -10), so the centre lands at (10, -10)
    assert centre(r.parts["B1"]) == (10.0, -10.0, 20.0)
    assert round(r.parts["B1"].volume, 3) == 16000.0
    assert "F2.moved" in r.body_names["B1"]


def test_turn_about_a_tilted_axis():
    d = Design()
    block(d)
    # 180 about the line through the origin along (1, 1, 0): v' = 2 (v.n) n - v; v = (0, 0, 20) has v.n = 0, so (0, 0, -20)
    apply(d, {"op": "add_feature", "kind": "move", "params": {"angle": 180, "dir_x": 1, "dir_y": 1, "dir_z": 0}})
    assert centre(built(d).parts["B1"]) == (0.0, 0.0, -20.0)


def test_turn_then_move_and_copy_into_a_new_body():
    d = Design()
    block(d)
    apply(d, {"op": "add_feature", "kind": "move", "params": {"move_x": 30, "move_z": 5, "into": "new"}})
    assert [b.id for b in d.bodies] == ["B1", "B2"] and d.get("F2").params["into"] == "B2"
    r = built(d)
    assert centre(r.parts["B1"]) == (0.0, 0.0, 20.0)  # the original stays
    assert centre(r.parts["B2"]) == (30.0, 0.0, 25.0)
    assert {b: round(s.volume, 3) for b, s in r.parts.items()} == {"B1": 16000.0, "B2": 16000.0}


def test_turn_about_a_picked_edge():
    d = Design()
    block(d)
    # the vertical edge at (10, 10): 180 about it takes the centre (0, 0) to (20, 20), whichever way the edge runs
    edge = {"at": [10, 10, 20], "tangent": [0, 0, 1], "length": 40}
    fid = apply(d, {"op": "add_feature", "kind": "move", "params": {"angle": 180}})
    apply(d, {"op": "edit_feature", "id": fid, "edges": [edge]})
    assert centre(built(d).parts["B1"]) == (20.0, 20.0, 20.0)


def test_a_zero_axis_direction_is_refused():
    d = Design()
    block(d)
    apply(d, {"op": "add_feature", "kind": "move", "params": {"angle": 30, "dir_z": 0}})
    r = build_design(d)
    assert not r.complete
    assert "direction is zero" in next(f.error for f in r.features if f.error)


def test_copy_into_the_same_body_is_refused():
    d = Design()
    block(d)
    apply(d, {"op": "add_feature", "kind": "move", "params": {"move_x": 5, "into": "B1"}})
    r = build_design(d)
    assert "different body" in next(f.error for f in r.features if f.error)


@pytest.mark.parametrize("angle", [0.0, 360.0])
def test_no_turn_leaves_the_body_where_it_was(angle):
    d = Design()
    block(d)
    apply(d, {"op": "add_feature", "kind": "move", "params": {"angle": angle, "axis_x": 50}})
    assert centre(built(d).parts["B1"]) == (0.0, 0.0, 20.0)
