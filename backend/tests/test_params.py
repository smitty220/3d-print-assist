import math

import pytest

from print_assist.features.model import Design
from print_assist.features.model import FeatureInstance as FI
from print_assist.params.expr import ExprError, evaluate, names
from print_assist.params.graph import resolve
from print_assist.params.model import Parameter as P
from print_assist.sketch.model import Constraint, Sketch


def test_formulas_evaluate_with_degrees_and_units():
    env = {"h": 120.0, "wall": 1.6}
    assert evaluate("2*h + sin(30)", env) == pytest.approx(240.5)
    assert evaluate("atan(1)") == pytest.approx(45.0)
    assert evaluate("atan2(1, 1)") == pytest.approx(45.0)
    assert evaluate("2 * inch") == pytest.approx(50.8)
    assert evaluate("max(wall, 2 * 0.4) if h > 100 else 0", env) == pytest.approx(1.6)
    assert evaluate("clamp(h, 10, 50)", env) == 50.0
    assert evaluate("68 * 2 / sqrt(3)") == pytest.approx(136 / math.sqrt(3))
    assert names("base_d * cos(30) + pi + h") == {"base_d", "h"}


@pytest.mark.parametrize("src", [
    "__import__('os')", "h.__class__", "(lambda: 1)()", "[1, 2]", "h[0]", "{'a': 1}", "open('x')",
    "max(*[1, 2])", "round(h, ndigits=1)", "'text'", "x if True else 1", "2 ** 1000", "1/0", "a = 1", "",
    "9" * 600,
])
def test_anything_else_is_refused(src):
    with pytest.raises(ExprError):
        evaluate(src, {"h": 1.0, "x": 1.0})


def test_parameters_resolve_in_dependency_order_with_transitive_users():
    sk = Sketch(id="S1", constraints=[Constraint(id="d", kind="diameter", refs=["c"], expr="base_d"),
                                      Constraint(id="a", kind="angle", refs=["l1", "l2"], expr="90 - 30")])
    d = Design(
        parameters=[P(name="top_d", expr="base_d * 1.4"), P(name="base_d", expr="68"), P(name="height", expr="120"),
                    P(name="label", quantity="text", expr="SAGE"), P(name="n", quantity="count", expr="5.6")],
        sketches=[sk],
        features=[FI(id="F1", kind="plate", exprs={"length": "top_d", "thickness": "height / 24"}),
                  FI(id="F2", kind="fillet", exprs={"radius": "1"})],
    )
    r = resolve(d)
    assert r.errors == {}
    assert r.values["top_d"] == pytest.approx(95.2) and r.values["label"] == "SAGE" and r.values["n"] == 6
    assert r.feature_values["F1"] == {"length": pytest.approx(95.2), "thickness": pytest.approx(5.0)}
    assert r.sketch_values["S1"]["d"] == pytest.approx(68) and r.sketch_values["S1"]["a"] == pytest.approx(math.pi / 3)
    # base_d drives F1 only through top_d, and S1 directly.
    assert r.users["base_d"] == {"param:top_d", "feature:F1", "sketch:S1"}
    assert r.inputs["feature:F1"] == {"top_d", "base_d", "height"}
    assert r.users["height"] == {"feature:F1"}


def test_cycles_unknown_names_and_bad_feature_params_are_errors_not_exceptions():
    d = Design(
        parameters=[P(name="a", expr="b + 1"), P(name="b", expr="a * 2"), P(name="c", expr="nope + 1"),
                    P(name="ok", expr="3")],
        features=[FI(id="F1", kind="plate", exprs={"length": "a", "width": "ok * 10", "bogus": "1"})],
    )
    r = resolve(d)
    assert "circular" in r.errors["param:a"] and "circular" in r.errors["param:b"]
    assert "unknown name 'nope'" in r.errors["param:c"]
    assert "a has an error" in r.errors["feature:F1.length"]
    assert "no parameter 'bogus'" in r.errors["feature:F1.bogus"]
    assert r.feature_values["F1"] == {"width": 30.0}
    assert r.error_for("feature:F1") and not r.error_for("param:ok")


def test_parameter_names_are_checked():
    for bad in ("2x", "sin", "pi", "in", "a-b", "__x"):
        with pytest.raises(ValueError):
            P(name=bad, expr="1")
