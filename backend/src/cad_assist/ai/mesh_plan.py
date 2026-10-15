"""Claude proposing FEA mesh controls from the engineer's best practices: it sees the part's edges and faces that
matter (pin bores, fillets, faces that could be mapped), sets controls, and every proposal is meshed and measured
(elements around and through each bore, element quality, size and memory) before it finishes. The chosen controls go
into the project for the engineer to check and edit before solving."""

from __future__ import annotations

import json

import numpy as np
from build123d import CenterOf, GeomType
from pydantic import ValidationError

from ..fem.analysis import AnalysisError
from ..fem.controls import EdgeDivision, MappedFace, MeshSetup, Sphere, edge_ref, face_ref, mappable, propose

SYSTEM = """You set up the finite-element mesh for a stress analysis inside CAD Assist.

The engineer is a career aerospace stress engineer. The part is a machined plate on pins (a bell crank, lever or \
fitting), analysed with quadratic tetrahedra (C3D10): pin forces go on the bores as cosine bearing pressure and the \
peak stresses sit at the bores (net section, bearing) and in fillets. Your job: choose mesh controls that follow the \
engineer's best practices (given below) at a sensible cost, check them, and explain them briefly. The engineer reviews \
and edits your controls before the solve, so keep them few and meaningful.

Units: millimetres. Controls (mesh_and_check's `setup`):
- size: the largest element edge; curvature: elements per full circle on curved faces (0 turns it off); min_size.
- edges: [{edge, n, label}] exactly n elements along an edge (by its id in the catalogue).
- faces: [{face, label}] mapped (structured) meshing on a face with one boundary of 3 or 4 edges.
- spheres: [{center: [x, y, z], radius, size, label}] element size `size` inside the sphere, blending to the global \
size outside.

Each check reports the mesh: tetrahedra, degrees of freedom and the memory to solve it, element quality (scaled \
Jacobian: 1 ideal, below 0.2 poor), elements around and through the wall at each bore, and any control Gmsh could not \
apply. Start from the rules proposal (given) unless the practices call for something else; change a few things per \
check, saying why in one short sentence. Use at most the checks allowed, then call finish with a short summary. Keep \
notes short: the engineer reads them live."""

TOOLS = [
    {
        "name": "mesh_and_check",
        "description": "Mesh the part with these controls and measure the mesh (no solve).",
        "input_schema": {
            "type": "object",
            "properties": {
                "note": {"type": "string", "description": "One sentence for the engineer: what this changes and why."},
                "setup": {"type": "object", "description": "size, curvature, min_size, edges, faces, spheres (see the system prompt)."},
            },
            "required": ["note", "setup"],
        },
    },
    {
        "name": "finish",
        "description": "End the session: the last mesh checked becomes the project's mesh for the engineer to review.",
        "input_schema": {
            "type": "object",
            "properties": {
                "summary": {"type": "string", "description": "2-3 sentences: the mesh and why, anything the engineer should look at."},
                "notes": {"type": "array", "items": {"type": "string"}, "description": "One line per control: why it is there."},
            },
            "required": ["summary"],
        },
    },
]


def _v(p) -> list[float]:
    return [round(p.X, 2), round(p.Y, 2), round(p.Z, 2)]


def catalogue(intent, part) -> str:
    """The entities worth controlling, by id: bores, circular edges, edges through the thickness, curved faces and faces
    that could be mapped."""
    bb = part.bounding_box()
    lines = [f"Part bounding box: {_v(bb.min)} to {_v(bb.max)} mm; volume {part.volume:.0f} mm^3."]
    bores = [o for o in intent.objects if o.kind == "bore"]
    for b in bores:
        lines.append(f"Pin {b.id}: diameter {b.diameter:.2f} mm, axis {[round(x, 3) for x in b.frame.z]} through "
                     f"{[round(x, 2) for x in b.frame.origin]}")
    axis = np.asarray(bores[0].frame.z, float) if bores else np.array([0, 0, 1.0])
    circ, thru = [], []
    for i, e in enumerate(part.edges()):
        if e.geom_type == GeomType.CIRCLE:
            circ.append(f"  e{i}: circular, radius {e.radius:.2f}, centre {_v(e.arc_center)}, length {e.length:.2f}")
        elif e.geom_type == GeomType.LINE:
            d = e.position_at(1) - e.position_at(0)
            if e.length > 1e-6 and abs(abs(np.dot([d.X, d.Y, d.Z], axis)) / e.length - 1) < 1e-4:
                thru.append(f"  e{i}: straight through the thickness, length {e.length:.2f}, at {_v(e.center(CenterOf.MASS))}")
    curved, flat = [], []
    for i, f in enumerate(part.faces()):
        if f.geom_type == GeomType.CYLINDER:
            curved.append(f"  f{i}: cylinder, area {f.area:.1f}, at {_v(f.center(CenterOf.MASS))}")
        elif f.geom_type == GeomType.PLANE and mappable(part, i) is None:
            flat.append(f"  f{i}: plane with {len(f.outer_wire().edges())} edges, area {f.area:.1f}, at {_v(f.center(CenterOf.MASS))}")
    for title, rows in (("Circular edges (bore rims, fillets)", circ), ("Edges through the thickness", thru),
                        ("Cylindrical faces (bores, fillets)", curved), ("Faces that could be mapped", flat)):
        lines.append(f"{title}: {len(rows)}")
        lines += rows[:80] + ([f"  ... {len(rows) - 80} more"] if len(rows) > 80 else [])
    return "\n".join(lines)


def to_setup(raw: dict, part, practices: str) -> MeshSetup:
    """Claude's controls (entities by id) as a MeshSetup with each entity's centre of mass and size."""
    E, F = part.edges(), part.faces()
    s = MeshSetup(size=float(raw.get("size", 3.0)), curvature=int(raw.get("curvature", 28)), min_size=float(raw.get("min_size", 0.15)),
                  source="ai", practices=practices)
    for e in raw.get("edges") or []:
        i = int(str(e["edge"]).lstrip("e"))
        if not 0 <= i < len(E):
            raise ValueError(f"No edge e{i}")
        s.edges.append(EdgeDivision(edge=i, n=int(e["n"]), label=str(e.get("label", "")), **edge_ref(part, i)))
    for f in raw.get("faces") or []:
        i = int(str(f["face"]).lstrip("f"))
        if not 0 <= i < len(F):
            raise ValueError(f"No face f{i}")
        why = mappable(part, i)
        if why:
            raise ValueError(f"f{i} cannot be mapped: {why}")
        s.faces.append(MappedFace(face=i, label=str(f.get("label", "")), **face_ref(part, i)))
    for sp in raw.get("spheres") or []:
        s.spheres.append(Sphere(center=[float(x) for x in sp["center"]], radius=float(sp["radius"]), size=float(sp["size"]),
                                label=str(sp.get("label", ""))))
    if not 0.2 <= s.size <= 50 or s.min_size > s.size:
        raise ValueError("size must be 0.2-50 mm and min_size at most size")
    if any(sp.size < 0.1 for sp in s.spheres):
        raise ValueError("sphere sizes below 0.1 mm would not fit in memory")
    return s


def _setup_brief(s: MeshSetup) -> dict:
    return {"size": s.size, "curvature": s.curvature, "min_size": s.min_size,
            "edges": [{"edge": f"e{e.edge}", "n": e.n, "label": e.label} for e in s.edges],
            "faces": [{"face": f"f{f.face}", "label": f.label} for f in s.faces],
            "spheres": [{"center": sp.center, "radius": sp.radius, "size": sp.size, "label": sp.label} for sp in s.spheres]}


def run_plan(job, project, build, client, model: str, instruction: str, max_meshes: int, runner) -> MeshSetup | None:
    """The session; logs into `job` (a FemJob). Saves the chosen controls to the project and keeps their mesh as the
    preview. Returns them (None if nothing meshed)."""
    intent = project.data.intent
    part = build.world
    practices = project.data.mesh.practices
    rules = propose(intent, part, "medium", practices)
    first = "\n\n".join([
        "Best practices (the engineer's):\n" + practices,
        "Part:\n" + catalogue(intent, part),
        "Rules proposal (what the best-practice rules alone give):\n" + json.dumps(_setup_brief(rules)),
        f"Current controls ({project.data.mesh.source}):\n" + json.dumps(_setup_brief(project.data.mesh)),
        f"You may run {max_meshes} mesh checks." + (f"\n\nThe engineer asks: {instruction}" if instruction.strip() else ""),
    ])
    messages = [{"role": "user", "content": first}]
    system = [{"type": "text", "text": SYSTEM, "cache_control": {"type": "ephemeral"}}]
    last: tuple[MeshSetup, object, dict] | None = None
    checks = 0
    job.say("read", f"Claude is planning the mesh from {len(practices.strip().splitlines())} best-practice lines")

    def check(args) -> tuple[str, bool]:
        nonlocal last, checks
        if checks >= max_meshes:
            return "Check limit reached. Call finish now.", True
        try:
            setup = to_setup(args.get("setup") or {}, part, practices)
        except (ValueError, KeyError, TypeError, ValidationError) as exc:
            return f"Invalid setup: {exc}", True
        checks += 1
        job.say("ai", f"Mesh {checks}: {(args.get('note') or '').strip()}")
        try:
            mesh, m = runner.make_mesh(job, project, build, setup)
        except AnalysisError as exc:
            return f"Meshing failed: {exc}", True
        last = (setup, mesh, m)
        q = m["quality"]
        out = [f"Mesh {checks} of {max_meshes}: {m['tets']:,} tetrahedra, {m['dof']:,} DOF, about {m['gb']:.1f} GB to solve.",
               f"Quality (scaled Jacobian): min {q['min']:.2f}, mean {q['mean']:.2f}, {q['below_0_2']} elements below 0.2; "
               f"histogram 0-1 in tenths {q['histogram']}."]
        out += [f"{b['id']}: {b['around']} elements around the bore, about {b['through']} through the {b['thickness']:.2f} mm wall"
                for b in m["bores"]]
        out += [f"Note: {n}" for n in m["notes"]]
        return "\n".join(out), False

    for _turn in range(max_meshes * 3 + 3):
        if job.cancel.is_set():
            job.say("skip", "Stopped")
            break
        resp = client.create(model=model, max_tokens=4000, system=system, tools=TOOLS, messages=messages)
        messages.append({"role": "assistant", "content": [_block(b) for b in resp.content]})
        results, done = [], False
        for blk in resp.content:
            if blk.type == "text" and blk.text.strip():
                job.say("ai", blk.text.strip())
            elif blk.type == "tool_use":
                if blk.name == "mesh_and_check":
                    text, err = check(blk.input or {})
                elif blk.name == "finish":
                    args = blk.input or {}
                    job.say("result", (args.get("summary") or "").strip())
                    if last is not None:
                        last[0].notes = [str(n) for n in args.get("notes") or []] or [(args.get("summary") or "").strip()]
                    text, err, done = "Session ended.", False, True
                else:
                    text, err = f"Unknown tool {blk.name!r}", True
                results.append({"type": "tool_result", "tool_use_id": blk.id, "content": text, "is_error": err})
        if done or not results:
            break
        messages.append({"role": "user", "content": results})
    if last is None:
        job.say("skip", "No mesh was checked; the controls are unchanged")
        return None
    setup, mesh, m = last
    project.data.mesh = setup
    project.save()
    runner._save_preview(project, runner.mesh_key(project), mesh, m)
    job.mesh_stats = m
    job.say("result", f"Kept Claude's controls: {len(setup.spheres)} sphere(s), {len(setup.edges)} edge division(s), "
                      f"{len(setup.faces)} mapped face(s); check the mesh, then solve")
    return setup


def _block(b) -> dict:
    if b.type == "text":
        return {"type": "text", "text": b.text}
    if b.type == "tool_use":
        return {"type": "tool_use", "id": b.id, "name": b.name, "input": b.input}
    return b.model_dump() if hasattr(b, "model_dump") else dict(b)
