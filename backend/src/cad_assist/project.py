"""A project on disk: imported mating parts (STEP) plus the intent model.

Layout:  <root>/parts/<name>.stp   and   <root>/project.json
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Literal

from build123d import Shape
from pydantic import BaseModel, Field

from .geometry.samples import write_samples
from .geometry.signature import find_face
from .geometry.step_io import read_step
from .features.context import FeatureError
from .features.design import BuildResult, build_design
from .features.model import Design
from .fem.controls import MeshSetup
from .intent import builders
from .intent.models import Intent, KeepOut, face_part
from .validate import Report, validate

Role = Literal["mating", "keepout"]

BUILDERS = {"planar_mount": builders.planar_mount, "bolt_pattern": builders.bolt_pattern, "bore": builders.bore}


class PartInfo(BaseModel):
    role: Role = "mating"


class ProjectFile(BaseModel):
    parts: dict[str, PartInfo] = Field(default_factory=dict)
    intent: Intent = Field(default_factory=Intent)
    design: Design = Field(default_factory=Design)
    mesh: MeshSetup = Field(default_factory=MeshSetup)  # FEA mesh controls


def safe_name(name: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("._")
    if not cleaned:
        raise ValueError("Part name is empty after removing unsupported characters")
    return cleaned[:80]


def default_role(name: str) -> Role:
    return "keepout" if re.search(r"envelope|keep.?out|stay.?out", name, re.I) else "mating"


class Project:
    def __init__(self, root: Path):
        self.root = root
        self.parts_dir = root / "parts"
        self.file = root / "project.json"
        self._cache: dict[str, tuple[float, Shape]] = {}
        self._build: tuple[str, BuildResult] | None = None
        self._report: tuple[str, Report] | None = None
        self.parts_dir.mkdir(parents=True, exist_ok=True)
        self.data = ProjectFile.model_validate_json(self.file.read_text()) if self.file.exists() else ProjectFile()

    def save(self) -> None:
        tmp = self.file.with_suffix(".tmp")
        tmp.write_text(self.data.model_dump_json(indent=2))
        tmp.replace(self.file)

    def seed_samples(self) -> None:
        if self.data.parts:
            return
        for path in write_samples(self.parts_dir):
            self._register(path.stem)
        self.save()

    def part_path(self, name: str) -> Path:
        return self.parts_dir / f"{name}.stp"

    def shape(self, name: str) -> Shape:
        path = self.part_path(name)
        if name not in self.data.parts or not path.exists():
            raise KeyError(name)
        mtime = path.stat().st_mtime
        cached = self._cache.get(name)
        if cached and cached[0] == mtime:
            return cached[1]
        shape = read_step(path)
        self._cache[name] = (mtime, shape)
        return shape

    def add_part(self, name: str, step_bytes: bytes) -> list[str]:
        """Store (or replace) a part. Returns warnings for intent objects whose faces could not be re-found."""
        name = safe_name(name)
        path = self.part_path(name)
        tmp = path.with_suffix(".upload")
        tmp.write_bytes(step_bytes)
        try:
            read_step(tmp)  # validate before replacing anything
        except Exception:
            tmp.unlink(missing_ok=True)
            raise
        tmp.replace(path)
        self._cache.pop(name, None)
        replaced = name in self.data.parts
        self._register(name)
        warnings = self.rematch(name) if replaced else []
        self.save()
        return warnings

    def _register(self, name: str) -> None:
        if name not in self.data.parts:
            role = default_role(name)
            self.data.parts[name] = PartInfo(role=role)
            if role == "keepout":
                self._add_keepout(name)

    def _add_keepout(self, name: str) -> None:
        intent = self.data.intent
        if not any(o.kind == "keep_out" and o.part == name for o in intent.objects):
            intent.objects.append(KeepOut(id=intent.next_id("keep_out"), label=name, part=name, source="typed"))

    def set_role(self, name: str, role: Role) -> None:
        self.data.parts[name].role = role
        intent = self.data.intent
        if role == "keepout":
            self._add_keepout(name)
        else:
            intent.objects = [o for o in intent.objects if not (o.kind == "keep_out" and o.part == name)]
        self.save()

    def remove_part(self, name: str) -> None:
        self.data.parts.pop(name, None)
        self._cache.pop(name, None)
        self.part_path(name).unlink(missing_ok=True)
        self.data.intent.objects = [
            o for o in self.data.intent.objects
            if not (face_part(o) == name or (o.kind == "keep_out" and o.part == name))
        ]
        self._drop_dangling_refs()
        self.save()

    def _drop_dangling_refs(self) -> None:
        ids = {o.id for o in self.data.intent.objects}
        for o in self.data.intent.objects:
            if getattr(o, "parent", None) not in ids | {None}:
                o.parent = None
            if o.kind == "load_case" and o.target not in ids:
                o.target = None

    def delete_intent(self, obj_id: str) -> None:
        self.data.intent.objects = [o for o in self.data.intent.objects if o.id != obj_id]
        self._drop_dangling_refs()
        self.save()

    def rematch(self, name: str) -> list[str]:
        """After a part revision, point every face reference on it at the matching face of the new shape."""
        shape = self.shape(name)
        warnings = []
        objs = self.data.intent.objects
        for i, o in enumerate(objs):
            if face_part(o) != name:
                continue
            idx = find_face(shape, o.face.signature)
            if idx is None:
                warnings.append(f"{o.id}: could not find its face on the revised {name}")
                continue
            try:
                fresh = BUILDERS[o.kind](self.data.intent, shape, name, idx)
            except builders.BuildError as exc:
                warnings.append(f"{o.id}: {exc}")
                continue
            keep = {"id", "label", "strength", "source", "support"} | ({"fastener", "fit", "parent"} if o.kind == "bolt_pattern" else set())
            objs[i] = fresh.model_copy(update={k: getattr(o, k) for k in keep})
        return warnings

    def _key(self) -> str:
        return self.data.model_dump_json()

    def build(self) -> BuildResult:
        """Build the current design, reusing the last result while intent and design are unchanged."""
        key = self._key()
        if self._build is None or self._build[0] != key:
            self._build = (key, build_design(self.data.intent, self.data.design))
        return self._build[1]

    def adopt_build(self, build: BuildResult) -> None:
        """Cache a build of the current design made elsewhere (the topology interpreter builds it as it goes)."""
        self._build = (self._key(), build)

    def validate(self) -> Report:
        """Pre-flight the intent and, when a design exists, check the built part. Cached like build()."""
        key = self._key()
        if self._report is None or self._report[0] != key:
            build = None
            if self.data.design.features:
                try:
                    build = self.build()
                except FeatureError:
                    pass  # no design frame yet; pre-flight reports the missing mount
            roles = {k: v.role for k, v in self.data.parts.items()}
            self._report = (key, validate(self.data.intent, build, self.shape, roles))
        return self._report[1]

    def summary(self) -> dict:
        return json.loads(self.data.model_dump_json())
