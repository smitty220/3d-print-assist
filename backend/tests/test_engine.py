"""The incremental build engine and the workspace's update states."""

import pytest

from print_assist.build.engine import Engine
from print_assist.build.ops import OpError
from print_assist.build.workspace import ChangeRefused, Workspace
from print_assist.features.model import Design
from print_assist.features.model import FeatureInstance as FI
from print_assist.params.model import Parameter as P
from print_assist.project import Projects
from print_assist.rules import PrintRules

RECT = [[0, 0], [40, 0], [40, 30], [0, 30]]
HOLE = [[5, 5], [15, 5], [15, 15], [5, 15]]
SLOT = [[25, 5], [35, 5], [35, 25], [25, 25]]


def design():
    """A 40 x 30 plate, `t` thick, with a 10 x 10 pocket `d` deep and a 10 x 20 pocket 1 deep."""
    return Design(
        parameters=[P(name="t", expr="10"), P(name="d", expr="4")],
        features=[
            FI(id="F1", kind="profile_plate", params={"outline": RECT, "corner_radius": 0.0}, exprs={"z_max": "t"}),
            FI(id="F2", kind="profile_cut", refs={"base": "F1"}, params={"outline": HOLE, "corner_radius": 0.0}, exprs={"depth": "d"}),
            FI(id="F3", kind="profile_cut", refs={"base": "F1"}, params={"outline": SLOT, "corner_radius": 0.0, "depth": 1.0}),
        ],
    )


def volume(t, d):
    """40·30·t − 10·10·d − 10·20·1"""
    return 40 * 30 * t - 100 * d - 200


def test_rebuilds_only_from_the_first_changed_feature():
    eng = Engine()
    d = design()
    r = eng.build(d, PrintRules())
    assert eng.builds == 3 and r.complete and r.volume == pytest.approx(volume(10, 4))
    d.parameters[1].expr = "6"  # d is used by F2 only: F1 comes from the cache
    r = eng.build(d, PrintRules())
    assert eng.builds == 5 and r.volume == pytest.approx(volume(10, 6))
    r = eng.build(d, PrintRules())  # nothing changed: nothing rebuilt
    assert eng.builds == 5 and r.volume == pytest.approx(volume(10, 6))
    d.parameters[0].expr = "12"  # t drives F1: everything after it too
    eng.build(d, PrintRules())
    assert eng.builds == 8


def test_a_failure_stops_the_build_and_skips_what_follows():
    eng = Engine()
    d = design()
    d.parameters[1].expr = "12"  # deeper than the plate
    r = eng.build(d, PrintRules())
    st = {f.id: f.status for f in r.features}
    assert st == {"F1": "ok", "F2": "fail", "F3": "skipped"} and not r.complete
    assert "far face" in r.features[1].error and "F2 failed" in r.features[2].message


def test_formula_out_of_bounds_is_a_failure_not_a_clamp():
    d = design()
    d.parameters[1].expr = "-1"
    r = Engine().build(d, PrintRules())
    assert r.features[1].status == "fail" and "must be at least" in r.features[1].error


def test_rollback_builds_up_to_the_bar():
    d = design()
    d.rollback = 1
    r = Engine().build(d, PrintRules())
    assert [f.status for f in r.features] == ["ok", "skipped", "skipped"]
    assert r.volume == pytest.approx(40 * 30 * 10)


def test_thin_feature_is_built_with_a_warning():
    d = design()
    d.parameters[0].expr = "0.6"
    d.features = d.features[:1]
    r = Engine().build(d, PrintRules())
    assert r.features[0].status == "warn" and "Thickness" in r.features[0].message and r.complete


@pytest.fixture
def ws(tmp_path):
    p = Projects(tmp_path).create("Test")
    p.data.design = design()
    p.save()
    return Workspace(p)


def test_manual_mode_marks_what_a_change_affects_then_updates(ws):
    t = ws.tree()
    assert t["pending_count"] == 0 and t["mode"] == "manual" and t["exportable"]
    builds = ws.engine.builds
    ws.change([{"op": "set_parameter", "name": "d", "expr": "5"}])
    t = ws.tree()
    status = {f["id"]: f["status"] for f in t["features"]}
    assert status == {"F1": "ok", "F2": "stale", "F3": "stale"}
    assert t["pending_count"] == 2 and not t["exportable"]
    assert next(p for p in t["params"] if p["name"] == "d")["status"] == "stale"
    assert ws.engine.builds == builds  # nothing built yet
    part = ws.part()
    assert part["stale"] and part["volume"] == pytest.approx(volume(10, 4))  # still the last update
    ws.update()
    t = ws.tree()
    assert t["pending_count"] == 0 and {f["status"] for f in t["features"]} == {"ok"}
    assert ws.engine.builds == builds + 2 and ws.part()["volume"] == pytest.approx(volume(10, 5))
    assert [h["message"] for h in ws.project.history()][:2] == ["Update", "Create design"]


def test_auto_mode_rebuilds_on_every_change(ws):
    ws.set_mode("auto")
    ws.change([{"op": "set_parameter", "name": "t", "expr": "8"}])
    assert ws.tree()["pending_count"] == 0 and ws.part()["volume"] == pytest.approx(volume(8, 4))


def test_failed_update_keeps_the_last_good_part(ws):
    ws.change([{"op": "set_parameter", "name": "d", "expr": "20"}])
    ws.update()
    t = ws.tree()
    assert t["failed"] and not t["exportable"]
    assert {f["id"]: f["status"] for f in t["features"]} == {"F1": "ok", "F2": "fail", "F3": "skipped"}
    assert ws.part()["volume"] == pytest.approx(volume(10, 4)) and ws.part()["failed"]


def test_parameter_users_and_feature_inputs_are_in_the_tree(ws):
    t = ws.tree()
    d = next(p for p in t["params"] if p["name"] == "d")
    assert d["users"] == ["feature:F2"]
    assert next(f for f in t["features"] if f["id"] == "F1")["inputs"] == ["t"]


def test_ai_changes_are_built_first_and_refused_if_they_break_the_part(ws):
    before = ws.design.model_dump_json()
    with pytest.raises(ChangeRefused, match="F2"):
        ws.change([{"op": "set_parameter", "name": "d", "expr": "30"}], actor="claude-code", message="deeper pocket")
    assert ws.design.model_dump_json() == before and not ws.activity[-1].ok
    ws.change([{"op": "set_parameter", "name": "d", "expr": "2"}], actor="claude-code", message="shallower")
    assert ws.tree()["pending_count"] == 0  # the AI's change was built, even in manual mode
    ws.change([{"op": "edit_feature", "id": "F1", "locked": True}])
    with pytest.raises(OpError, match="locked"):
        ws.change([{"op": "edit_feature", "id": "F1", "params": {"z_min": 1.0}}], actor="claude-code")
    with pytest.raises(ChangeRefused, match="changed since"):
        ws.change([{"op": "set_parameter", "name": "d", "expr": "3"}], base_rev=0)


def test_ops_validate_their_input(ws):
    for op, msg in [
        ({"op": "add_feature", "kind": "nope"}, "unknown feature kind"),
        ({"op": "add_feature", "kind": "plate", "params": {"bogus": 1}}, "no parameter"),
        ({"op": "add_feature", "kind": "profile_cut", "refs": {"base": "F9"}}, "not a sketch or an earlier feature"),
        ({"op": "delete_feature", "id": "F1"}, "used by F2, F3"),
        ({"op": "delete_parameter", "name": "t"}, "used by F1"),
        ({"op": "set_parameter", "name": "t", "expr": "d + "}, "cannot read"),
        ({"op": "set_parameter", "name": "d", "expr": "e2 * 2", "quantity": "length"}, None),
        ({"op": "frobnicate"}, "unknown op"),
        ({"op": "edit_feature"}, "needs 'id'"),
    ]:
        if msg is None:
            continue
        with pytest.raises(OpError, match=msg):
            ws.change([op])
    made = ws.change([{"op": "set_parameter", "name": "w", "expr": "t * 2"},
                      {"op": "add_feature", "kind": "plate", "exprs": {"length": "w"}, "params": {"center_x": 100}}])
    assert made == ["w", "F4"] and ws.design.get("F4").exprs == {"length": "w"}
