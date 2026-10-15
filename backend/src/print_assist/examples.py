"""Ready-made example projects, set up the way a user would pick them.

    uv run python -m print_assist.examples bell_crank data/projects/bell_crank
"""

from __future__ import annotations

import sys
from pathlib import Path

from build123d import IN, GeomType

from .geometry.samples import BELL_CRANK, write_samples
from .intent import builders
from .intent.models import DesignSpace, Frame, LoadCase
from .project import Project

LBF = 4.4482216  # N


def _bore_face(shape, radius: float) -> int:
    for i, f in enumerate(shape.faces()):
        if f.geom_type == GeomType.CYLINDER and abs(f.radius - radius) < 1e-3:
            return i
    raise ValueError("no bore of that radius")


def bell_crank(project: Project, input_lbf: float = 1500.0) -> None:
    """Bell crank: pivot (Pin) at the origin, input rod end 4.0 in along +X pulling `input_lbf` (limit) along -Y,
    output link 2.5 in along +Y reacting along its rod (X). Pins along Z; the crank may be up to 1.0 in thick."""
    if project.data.parts:
        raise ValueError(f"{project.root} already has parts; start from an empty project")
    for path in write_samples(project.parts_dir, BELL_CRANK):
        project._register(path.stem)
    intent = project.data.intent
    support = {"pivot_fitting": "pin", "input_rod_end": "free", "output_link_end": "link"}
    label = {"pivot_fitting": "Pivot", "input_rod_end": "Input rod end", "output_link_end": "Output link"}
    ids = {}
    for name in BELL_CRANK:
        shape = project.shape(name)
        b = builders.bore(intent, shape, name, _bore_face(shape, 0.377 / 2 * IN))
        b.support = support[name]
        b.label = f"{label[name]}: {b.label}"
        intent.objects.append(b)
        ids[name] = b.id
    world = Frame(origin=(0.0, 0.0, 0.0), z=(0.0, 0.0, 1.0), x=(1.0, 0.0, 0.0))
    intent.objects.append(DesignSpace(
        id=intent.next_id("design_space"), label="Design space", source="typed", frame=world,
        min=(-0.75 * IN, -0.75 * IN, -0.5 * IN), max=(4.75 * IN, 3.25 * IN, 0.5 * IN),
    ))
    intent.objects.append(LoadCase(
        id=intent.next_id("load_case"), label="Input rod pull", source="typed", target=ids["input_rod_end"],
        force=(0.0, -input_lbf * LBF, 0.0),
    ))
    project.save()


EXAMPLES = {"bell_crank": bell_crank}

if __name__ == "__main__":
    name, root = sys.argv[1], Path(sys.argv[2])
    EXAMPLES[name](Project(root))
    print(f"{name} written to {root}")
