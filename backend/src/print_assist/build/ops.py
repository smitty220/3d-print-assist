"""Design operations: every change to a design, as data. The REST endpoints, the AI chat and the MCP server all change
designs through these, so a change means the same thing whoever makes it.

An op is a dict with an "op" key. `apply(design, op, actor)` changes the design in place and returns what the op made
(a new id) or None; it raises OpError for anything invalid. Locked features refuse changes from any actor but the
user.
"""

from __future__ import annotations

import re
from typing import Any

from pydantic import ValidationError

from ..features.library import FEATURES
from ..features.model import Body, Colorway, Design, EdgePick, FeatureInstance, Filament, Reference, Variant
from ..params.expr import ExprError, is_literal, names
from ..params.graph import resolve
from ..params.model import Parameter
from ..sketch.model import Constraint, FacePick, Placement, Sketch
from ..sketch.solve import SketchSolveError, solve

USER = "user"


class OpError(ValueError):
    pass


def _feature(d: Design, fid: str) -> FeatureInstance:
    f = d.get(fid)
    if f is None:
        raise OpError(f"no feature {fid!r}")
    return f


def _sketch(d: Design, sid: str) -> Sketch:
    s = d.sketch(sid)
    if s is None:
        raise OpError(f"no sketch {sid!r}")
    return s


def _guard(f: FeatureInstance, actor: str) -> None:
    if f.locked and actor != USER:
        raise OpError(f"{f.id} is locked by the user; leave it as it is")


def _check_refs(d: Design, refs: dict[str, str], before: int) -> None:
    earlier = {f.id for f in d.features[:before]}
    sketches = {s.id for s in d.sketches}
    for role, v in refs.items():
        if v and v not in earlier and v not in sketches and not ("." in v and v.split(".")[0] in earlier):
            raise OpError(f"ref {role} = {v!r} is not a sketch or an earlier feature")


def _split_values(kind: str, values: dict[str, Any]) -> tuple[dict, dict]:
    """Literal values and formulas: a string given for a number is a formula (`"height * 0.8"`)."""
    if not isinstance(values, dict):
        raise OpError(f"{kind} params must be a dictionary, got {type(values).__name__}")
    spec = {s.name: s for s in FEATURES[kind].params}
    bad = sorted(set(values) - set(spec))
    if bad:
        raise OpError(f"{kind} has no parameter(s) {', '.join(bad)}; it has {', '.join(spec)}")
    lit, exprs = {}, {}
    for k, v in values.items():
        numeric = spec[k].quantity in ("length", "ratio", "angle", "count")
        if numeric and isinstance(v, str) and not is_literal(v):
            try:
                names(v)
            except ExprError as exc:
                raise OpError(f"{k}: {exc}") from exc
            exprs[k] = v
        elif spec[k].quantity == "text" and isinstance(v, str) and v.startswith("="):
            exprs[k] = v[1:]
        elif numeric and isinstance(v, str):
            lit[k] = float(v)
        else:
            lit[k] = v
    return lit, exprs


BODY_PARAMS = ("into", "tool")  # text parameters that name a body
HEX = re.compile(r"^#[0-9a-fA-F]{6}$")


def _body_params(d: Design, kind: str, fid: str, lit: dict, adding: bool, name: str | None) -> None:
    """Check the parameters that name a body; "new" (Split's default) makes a new body for the feature."""
    for spec in FEATURES[kind].params:
        if spec.name not in BODY_PARAMS:
            continue
        v = lit.get(spec.name, spec.default if adding else None)
        if v == "new":
            bid = d.next_body_id()
            d.bodies.append(Body(id=bid, name=name or f"{FEATURES[kind].label} {fid}"))
            lit[spec.name] = bid
        elif v and d.body(v) is None:
            raise OpError(f"{spec.name}: no body {v!r}; bodies are {', '.join(b.id for b in d.bodies)} (or \"new\")")


def _color(v: Any) -> str:
    if not isinstance(v, str) or not HEX.match(v):
        raise OpError(f"colour {v!r} must be #rrggbb")
    return v.lower()


def _solve_sketch(d: Design, sk: Sketch) -> Sketch:
    """Solve a sketch with its formula-driven dimensions filled in from the design's parameters."""
    trial = d.model_copy(update={"sketches": [sk if s.id == sk.id else s for s in d.sketches]})
    if not any(s.id == sk.id for s in d.sketches):
        trial.sketches = [*d.sketches, sk]
    r = resolve(trial)
    errs = r.error_for(f"sketch:{sk.id}")
    if errs:
        raise OpError(errs[0])
    for c in sk.constraints:
        if c.id in r.sketch_values.get(sk.id, {}):
            c.value = r.sketch_values[sk.id][c.id]
    if not sk.entities:
        return sk
    try:
        out = solve(sk)
    except (SketchSolveError, ValueError) as exc:
        raise OpError(f"{sk.id}: {exc}") from exc
    out.placement, out.after = sk.placement, sk.after
    return out


def apply(d: Design, op: dict, actor: str = USER) -> str | None:
    kind = op.get("op")
    if kind not in _OPS:
        raise OpError(f"unknown op {kind!r}; ops are {', '.join(_OPS)}")
    try:
        return _OPS[kind](d, op, actor)
    except KeyError as exc:
        raise OpError(f"{kind} needs {exc}") from None
    except ValidationError as exc:
        raise OpError(str(exc.errors()[0]["msg"])) from None


# ---------------------------------------------------------------- parameters


def _set_parameter(d: Design, op: dict, actor: str):
    name = op["name"]
    expr = str(op["expr"]).strip()
    p = d.parameter(name)
    quantity = op.get("quantity") or (p.quantity if p else "length")
    if quantity != "text" and not is_literal(expr):
        try:
            names(expr)
        except ExprError as exc:
            raise OpError(f"{name}: {exc}") from exc
    if p is None:
        d.parameters.append(Parameter(name=name, quantity=quantity, expr=expr, comment=op.get("comment", "")))
    else:
        p.expr, p.quantity = expr, quantity
        if "comment" in op:
            p.comment = op["comment"]
    r = resolve(d)
    if f"param:{name}" in r.errors and "circular" in r.errors[f"param:{name}"]:
        raise OpError(f"{name}: {r.errors[f'param:{name}']}")
    return name


def _delete_parameter(d: Design, op: dict, actor: str):
    name = op["name"]
    if d.parameter(name) is None:
        raise OpError(f"no parameter {name!r}")
    users = sorted(resolve(d).users.get(name, set()))
    if users:
        raise OpError(f"{name} is used by {', '.join(u.split(':', 1)[1] for u in users)}")
    d.parameters = [p for p in d.parameters if p.name != name]
    return None


# ---------------------------------------------------------------- features


def _add_feature(d: Design, op: dict, actor: str):
    kind = op["kind"]
    if kind not in FEATURES:
        raise OpError(f"unknown feature kind {kind!r}; kinds are {', '.join(FEATURES)}")
    lit, exprs = _split_values(kind, {**op.get("params", {}), **op.get("exprs", {})})
    fid = op.get("id") or d.next_id()
    if d.get(fid):
        raise OpError(f"feature id {fid} is taken")
    body = op.get("body", "B1")
    if d.body(body) is None:
        raise OpError(f"no body {body!r}")
    index = op.get("index", len(d.features) if d.rollback is None else d.rollback)
    refs = op.get("refs", {})
    _check_refs(d, refs, index)
    _body_params(d, kind, fid, lit, True, op.get("new_body_name"))
    f = FeatureInstance(id=fid, kind=kind, params=lit, exprs=exprs, refs=refs, body=body,
                        faces=[FacePick.model_validate(x) for x in op.get("faces", [])],
                        locked=bool(op.get("locked", False)) and actor == USER)
    d.features.insert(index, f)
    if d.rollback is not None and index <= d.rollback:
        d.rollback += 1
    return fid


def _edit_feature(d: Design, op: dict, actor: str):
    f = _feature(d, op["id"])
    _guard(f, actor)
    if "params" in op or "exprs" in op:
        lit, exprs = _split_values(f.kind, {**op.get("params", {}), **op.get("exprs", {})})
        _body_params(d, f.kind, f.id, lit, False, op.get("new_body_name"))
        for k in lit:
            f.exprs.pop(k, None)
        f.params = f.params | lit
        for k, v in exprs.items():
            f.exprs[k] = v
    if "refs" in op:
        _check_refs(d, op["refs"], d.features.index(f))
        f.refs = f.refs | op["refs"]
    if "edges" in op:
        f.edges = [EdgePick.model_validate(e) for e in op["edges"]]
    if "faces" in op:
        f.faces = [FacePick.model_validate(x) for x in op["faces"]]
    if "body" in op:
        if d.body(op["body"]) is None:
            raise OpError(f"no body {op['body']!r}")
        f.body = op["body"]
    if "suppressed" in op:
        f.suppressed = bool(op["suppressed"])
    if "locked" in op:
        if actor != USER:
            raise OpError("only the user can lock or unlock a feature")
        f.locked = bool(op["locked"])
    return None


def _delete_feature(d: Design, op: dict, actor: str):
    f = _feature(d, op["id"])
    _guard(f, actor)
    users = [g.id for g in d.features if any(v == f.id or v.startswith(f.id + ".") for v in g.refs.values())]
    if users:
        raise OpError(f"{f.id} is used by {', '.join(users)}; change or delete those first")
    i = d.features.index(f)
    d.features.remove(f)
    if d.rollback is not None and i < d.rollback:
        d.rollback -= 1
    return None


def _move_feature(d: Design, op: dict, actor: str):
    f = _feature(d, op["id"])
    _guard(f, actor)
    index = int(op["index"])
    if not 0 <= index < len(d.features):
        raise OpError(f"index {index} is outside the tree (0 to {len(d.features) - 1})")
    before = list(d.features)
    d.features.remove(f)
    d.features.insert(index, f)
    for i, g in enumerate(d.features):  # a feature may only refer to features before it
        try:
            _check_refs(d, g.refs, i)
        except OpError:
            d.features = before
            raise OpError(f"moving {f.id} there would put {g.id} before what it uses") from None
    return None


def _set_rollback(d: Design, op: dict, actor: str):
    index = op.get("index")
    if index is not None and not 0 <= int(index) <= len(d.features):
        raise OpError(f"rollback index {index} is outside the tree")
    d.rollback = None if index is None or int(index) >= len(d.features) else int(index)
    return None


# ---------------------------------------------------------------- sketches


def _add_sketch(d: Design, op: dict, actor: str):
    sid = op.get("id") or d.next_sketch_id()
    if d.sketch(sid):
        raise OpError(f"sketch id {sid} is taken")
    sk = Sketch(id=sid, name=op.get("name") or f"Sketch{sid[1:]}", entities=op.get("entities", []),
                constraints=op.get("constraints", []), placement=Placement.model_validate(op.get("placement", {})),
                after=op.get("after"))
    d.sketches.append(_solve_sketch(d, sk))
    return sid


def _edit_sketch(d: Design, op: dict, actor: str):
    old = _sketch(d, op["id"])
    for f in d.features:
        if old.id in f.refs.values():
            _guard(f, actor)
    sk = Sketch(id=old.id, name=op.get("name", old.name), entities=op.get("entities", [e.model_dump() for e in old.entities]),
                constraints=[Constraint.model_validate(c) for c in op.get("constraints", [c.model_dump() for c in old.constraints])],
                placement=Placement.model_validate(op["placement"]) if "placement" in op else old.placement,
                after=op.get("after", old.after))
    if "entities" in op or "constraints" in op or any(c.expr for c in sk.constraints):
        sk = _solve_sketch(d, sk)
    else:
        sk.report = old.report
    d.sketches = [sk if s.id == old.id else s for s in d.sketches]
    return None


def _delete_sketch(d: Design, op: dict, actor: str):
    sk = _sketch(d, op["id"])
    users = [f.id for f in d.features if sk.id in f.refs.values()]
    if users:
        raise OpError(f"{sk.id} is used by {', '.join(users)}; change or delete those first")
    d.sketches = [s for s in d.sketches if s.id != sk.id]
    return None


# ---------------------------------------------------------------- bodies


def _body(d: Design, bid: str) -> Body:
    b = d.body(bid)
    if b is None:
        raise OpError(f"no body {bid!r}")
    return b


def _add_body(d: Design, op: dict, actor: str):
    bid = d.next_body_id()
    d.bodies.append(Body(id=bid, name=op.get("name") or f"Body {bid[1:]}"))
    _edit_body(d, {**op, "id": bid}, actor)
    return bid


def _edit_body(d: Design, op: dict, actor: str):
    b = _body(d, op["id"])
    if "name" in op:
        b.name = str(op["name"]).strip() or b.name
    if "color" in op:
        b.color = None if op["color"] is None else _color(op["color"])
    if "filament" in op:
        slot = op["filament"]
        if slot is not None and d.filament(int(slot)) is None:
            raise OpError(f"no filament in slot {slot}; the palette has {', '.join(str(f.slot) for f in d.filaments) or 'none'}")
        b.filament = None if slot is None else int(slot)
    return None


def _delete_body(d: Design, op: dict, actor: str):
    b = _body(d, op["id"])
    if len(d.bodies) == 1:
        raise OpError("a design keeps at least one body")
    users = [f.id for f in d.features if f.body == b.id or any(f.params.get(k) == b.id for k in BODY_PARAMS)]
    if users:
        raise OpError(f"{b.id} is used by {', '.join(users)}; change or delete those first")
    d.bodies.remove(b)
    return None


# ---------------------------------------------------------------- filaments, variants, colorways


def _set_filament(d: Design, op: dict, actor: str):
    """Add or change the palette's filament in a slot."""
    slot = int(op["slot"])
    if slot < 1:
        raise OpError("slots are numbered from 1")
    color = _color(op["color"]) if "color" in op else None
    f = d.filament(slot)
    if f is None:
        f = Filament(slot=slot)
        d.filaments.append(f)
        d.filaments.sort(key=lambda x: x.slot)
    for k in ("name", "material"):
        if k in op:
            setattr(f, k, str(op[k]))
    if color:
        f.color = color
    if "nozzle" in op:
        f.nozzle = int(op["nozzle"])
    return str(slot)


def _delete_filament(d: Design, op: dict, actor: str):
    slot = int(op["slot"])
    if d.filament(slot) is None:
        raise OpError(f"no filament in slot {slot}")
    d.filaments = [f for f in d.filaments if f.slot != slot]
    for b in d.bodies:
        if b.filament == slot:
            b.filament = None
    return None


def _next(prefix: str, used: list[str]) -> str:
    n = 1
    while f"{prefix}{n}" in used:
        n += 1
    return f"{prefix}{n}"


def _set_variant(d: Design, op: dict, actor: str):
    """Add or change a variants-table row: a name and the parameter values that differ from the design's."""
    overrides = None
    if "overrides" in op:
        overrides = {}
        for name, expr in op["overrides"].items():
            expr = str(expr).strip()
            if expr == "":
                continue
            p = d.parameter(name)
            if p is None:
                raise OpError(f"no parameter {name!r} to vary")
            if p.quantity != "text" and not is_literal(expr):
                try:
                    names(expr)
                except ExprError as exc:
                    raise OpError(f"{op.get('name', name)}, {name}: {exc}") from exc
            overrides[name] = expr
    vid = op.get("id") or _next("V", [v.id for v in d.variants])
    v = d.variant(vid)
    if v is None:
        v = Variant(id=vid, name=op.get("name") or vid)
        d.variants.append(v)
    if "name" in op:
        v.name = str(op["name"]).strip() or v.name
    if overrides is not None:
        v.overrides = overrides
    return vid


def _set_colorway(d: Design, op: dict, actor: str):
    """Add or change a colorway: filament colours (and names) per palette slot."""
    colors = {int(k): _color(v) for k, v in op["colors"].items() if v} if "colors" in op else None
    cid = op.get("id") or _next("C", [c.id for c in d.colorways])
    c = d.colorway(cid)
    if c is None:
        c = Colorway(id=cid, name=op.get("name") or cid)
        d.colorways.append(c)
    if "name" in op:
        c.name = str(op["name"]).strip() or c.name
    if colors is not None:
        c.colors = colors
    if "names" in op:
        c.names = {int(k): str(v) for k, v in op["names"].items() if v}
    return cid


def _delete_variant(d: Design, op: dict, actor: str):
    if d.variant(op["id"]) is None:
        raise OpError(f"no variant {op['id']!r}")
    d.variants = [v for v in d.variants if v.id != op["id"]]
    return None


def _delete_colorway(d: Design, op: dict, actor: str):
    if d.colorway(op["id"]) is None:
        raise OpError(f"no colorway {op['id']!r}")
    d.colorways = [c for c in d.colorways if c.id != op["id"]]
    return None


# ---------------------------------------------------------------- references

REF_FIELDS = ("name", "plate", "visible", "place", "x", "y", "z", "rotate_z", "scale", "color")


def _set_reference(d: Design, op: dict, actor: str):
    """Add a reference (a `file` in the references folder) or change one (`id`): name, plate, visibility, placement,
    colour."""
    rid = op.get("id")
    if rid:
        r = d.reference(rid)
        if r is None:
            raise OpError(f"no reference {rid!r}; references are {', '.join(x.id for x in d.references) or 'none'}")
        data = r.model_dump()
    else:
        f = str(op.get("file", "")).strip()
        if not f or "/" in f or "\\" in f or f.startswith("."):
            raise OpError("a reference needs the name of a file in the design's references folder")
        rid = _next("R", [x.id for x in d.references])
        data = {"id": rid, "name": op.get("name") or f.rsplit(".", 1)[0], "file": f}
    for k in REF_FIELDS:
        if k in op:
            data[k] = _color(op[k]) if k == "color" else op[k]
    try:
        new = Reference.model_validate(data)
    except ValidationError as exc:
        raise OpError(f"reference: {exc.errors()[0]['loc'][-1]}: {exc.errors()[0]['msg']}") from exc
    if new.scale <= 0 or (new.plate is not None and new.plate < 1):
        raise OpError("reference: scale must be positive and plates count from 1")
    d.references = [new if x.id == rid else x for x in d.references] if d.reference(rid) else [*d.references, new]
    return rid


def _delete_reference(d: Design, op: dict, actor: str):
    if d.reference(op["id"]) is None:
        raise OpError(f"no reference {op['id']!r}")
    d.references = [r for r in d.references if r.id != op["id"]]
    return None


_OPS = {
    "set_parameter": _set_parameter, "delete_parameter": _delete_parameter,
    "add_feature": _add_feature, "edit_feature": _edit_feature, "delete_feature": _delete_feature,
    "move_feature": _move_feature, "set_rollback": _set_rollback,
    "add_sketch": _add_sketch, "edit_sketch": _edit_sketch, "delete_sketch": _delete_sketch,
    "add_body": _add_body, "edit_body": _edit_body, "delete_body": _delete_body,
    "set_filament": _set_filament, "delete_filament": _delete_filament,
    "set_variant": _set_variant, "delete_variant": _delete_variant,
    "set_colorway": _set_colorway, "delete_colorway": _delete_colorway,
    "set_reference": _set_reference, "delete_reference": _delete_reference,
}
OP_NAMES = list(_OPS)
