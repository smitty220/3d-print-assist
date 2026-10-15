"""Resolve a design's parameters and the formulas that use them.

Parameters are evaluated in dependency order (a cycle is an error on every parameter in it). Feature parameters and
sketch dimensions bound to a formula are then evaluated against them. Alongside the values comes the dependency
graph both ways: which features and sketches each parameter drives (directly or through other parameters), and which
parameters each feature or sketch reads. Node keys: "param:<name>", "feature:<id>", "sketch:<id>".
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from .expr import ExprError, evaluate, is_literal, names

if TYPE_CHECKING:
    from ..features.model import Design


@dataclass
class Resolved:
    values: dict[str, float | str] = field(default_factory=dict)  # parameter -> value (text parameters: their text)
    errors: dict[str, str] = field(default_factory=dict)  # node key -> what is wrong
    feature_values: dict[str, dict[str, Any]] = field(default_factory=dict)  # feature id -> param -> value
    sketch_values: dict[str, dict[str, float]] = field(default_factory=dict)  # sketch id -> constraint id -> value
    sketch_offsets: dict[str, float] = field(default_factory=dict)  # sketch id -> placement offset from its formula
    users: dict[str, set[str]] = field(default_factory=dict)  # parameter -> node keys it drives (transitive)
    inputs: dict[str, set[str]] = field(default_factory=dict)  # node key -> parameters it reads (transitive)

    def error_for(self, prefix: str) -> list[str]:
        return [m for k, m in self.errors.items() if k == prefix or k.startswith(prefix + ".")]


def _text_param(design: "Design", src: str):
    p = design.parameter(src.strip().lstrip("="))
    return p if p is not None and p.quantity == "text" else None


def resolve(design: "Design") -> Resolved:
    from ..features.library import FEATURES

    r = Resolved()
    params = {p.name: p for p in design.parameters}

    # ---- parameters, in dependency order
    deps: dict[str, set[str]] = {}
    for name, p in params.items():
        if p.quantity == "text" or is_literal(p.expr):
            deps[name] = set()
            continue
        try:
            deps[name] = names(p.expr)
        except ExprError as exc:
            r.errors[f"param:{name}"] = str(exc)
            deps[name] = set()
    for name, ds in deps.items():
        unknown = sorted(d for d in ds if d not in params)
        if unknown and f"param:{name}" not in r.errors:
            r.errors[f"param:{name}"] = f"unknown name {unknown[0]!r}"

    order: list[str] = []
    indeg = {n: len([d for d in ds if d in params]) for n, ds in deps.items()}
    ready = [n for n, k in indeg.items() if k == 0]
    while ready:
        n = ready.pop()
        order.append(n)
        for m, ds in deps.items():
            if n in ds:
                indeg[m] -= 1
                if indeg[m] == 0:
                    ready.append(m)
    for n in params:
        if n not in order:
            r.errors[f"param:{n}"] = "circular: it depends on itself through " + \
                ", ".join(sorted(d for d in deps[n] if d not in order))

    for n in order:
        p = params[n]
        if f"param:{n}" in r.errors:
            continue
        if p.quantity == "text":
            r.values[n] = p.expr
            continue
        bad = [d for d in deps[n] if d not in r.values]
        if bad:
            r.errors[f"param:{n}"] = f"depends on {bad[0]}, which has an error"
            continue
        try:
            v = evaluate(p.expr, r.values)
            if p.quantity == "count":
                v = float(round(v))
            r.values[n] = v
        except ExprError as exc:
            r.errors[f"param:{n}"] = str(exc)

    # ---- direct reads of every node
    direct: dict[str, set[str]] = {f"param:{n}": {d for d in ds if d in params} for n, ds in deps.items()}

    def number(key: str, src: str, angle_rad: bool = False) -> float | None:
        try:
            reads = names(src)
        except ExprError as exc:
            r.errors[key] = str(exc)
            return None
        direct.setdefault(key.split(".")[0], set()).update(d for d in reads if d in params)
        try:
            v = evaluate(src, r.values)
        except ExprError as exc:
            missing = [d for d in reads if d in params and d not in r.values]
            r.errors[key] = f"{missing[0]} has an error" if missing else str(exc)
            return None
        return math.radians(v) if angle_rad else v

    for f in design.features:
        spec = {s.name: s for s in FEATURES[f.kind].params} if f.kind in FEATURES else {}
        node = f"feature:{f.id}"
        direct.setdefault(node, set())
        vals: dict[str, Any] = {}
        for pname, src in f.exprs.items():
            key = f"{node}.{pname}"
            s = spec.get(pname)
            if s is None:
                r.errors[key] = f"{f.kind} has no parameter {pname!r}"
                continue
            if s.quantity == "text":
                tp = _text_param(design, src)
                if tp is None:
                    r.errors[key] = f"no text parameter {src.strip().lstrip('=')!r}"
                    continue
                direct[node].add(tp.name)
                vals[pname] = r.values.get(tp.name, tp.expr)
                continue
            v = number(key, src)
            if v is not None:
                vals[pname] = round(v) if s.quantity == "count" else v
        r.feature_values[f.id] = vals

    for sk in design.sketches:
        node = f"sketch:{sk.id}"
        direct.setdefault(node, set())
        vals: dict[str, float] = {}
        for c in sk.constraints:
            if c.expr:
                v = number(f"{node}.{c.id}", c.expr, angle_rad=(c.kind == "angle"))
                if v is not None:
                    vals[c.id] = v
        r.sketch_values[sk.id] = vals
        if sk.placement.offset_expr:
            v = number(f"{node}.offset", sk.placement.offset_expr)
            if v is not None:
                r.sketch_offsets[sk.id] = v

    # ---- transitive closure both ways
    def reads(node: str, seen: set[str]) -> set[str]:
        out: set[str] = set()
        for p in direct.get(node, ()):
            if p in seen:
                continue
            seen.add(p)
            out.add(p)
            out |= reads(f"param:{p}", seen)
        return out

    for node in direct:
        r.inputs[node] = reads(node, set())
    for n in params:
        r.users[n] = {node for node, ins in r.inputs.items() if n in ins and node != f"param:{n}"}
    return r
