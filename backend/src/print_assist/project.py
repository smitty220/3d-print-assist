"""Designs on disk. Each design is a folder with its own git history:

    <data root>/<slug>/project.json     the model (parameters, sketches, features) and its settings
    <data root>/<slug>/fonts/           fonts the design uses (copied in, so it rebuilds anywhere)
    <data root>/<slug>/references/      reference bodies (STEP / STL)
    <data root>/<slug>/exports/         exported files (not tracked)

Every kept change is a git commit (`checkpoint`), so history and undo come from git.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from .features.design import BuildResult, build_design
from .features.model import Design
from .printers import DEFAULT_PRINTER, PrinterProfile, profiles
from .rules import PrintRules
from .validate import Report, validate

DEFAULT_ROOT = Path.home() / "Projects" / "3DPrinting" / "designs"


def data_root() -> Path:
    return Path(os.environ.get("PRINT_ASSIST_DATA", DEFAULT_ROOT)).expanduser()


class Settings(BaseModel):
    update_mode: Literal["manual", "auto"] = "manual"
    printer: str = DEFAULT_PRINTER
    slicer: str = ""
    ai_mode: Literal["mcp", "api"] = "mcp"
    ai_provider: Literal["auto", "agy", "gemini", "anthropic"] = "auto"
    ai_model: str = ""


class ProjectFile(BaseModel):
    name: str = "Untitled"
    design: Design = Field(default_factory=Design)
    rules: PrintRules = Field(default_factory=PrintRules)
    settings: Settings = Field(default_factory=Settings)


def slugify(name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
    if not slug:
        raise ValueError("the name needs at least one letter or digit")
    return slug[:60]


def _git(root: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=root, capture_output=True, text=True, encoding="utf-8", timeout=30)


class Project:
    def __init__(self, root: Path):
        self.root = root
        self.slug = root.name
        self.file = root / "project.json"
        for sub in ("fonts", "references", "exports"):
            (root / sub).mkdir(parents=True, exist_ok=True)
        self.data = ProjectFile.model_validate_json(self.file.read_text(encoding="utf-8")) if self.file.exists() else ProjectFile()
        self._build: tuple[str, BuildResult] | None = None
        self._report: tuple[str, Report] | None = None

    # ---------- storage ----------

    def save(self) -> None:
        tmp = self.file.with_suffix(".tmp")
        tmp.write_text(self.data.model_dump_json(indent=2), encoding="utf-8")
        tmp.replace(self.file)

    def init_git(self) -> None:
        if (self.root / ".git").exists():
            return
        (self.root / ".gitignore").write_text("exports/\n*.tmp\n.recovery.json\nthumbnail.png\n", encoding="utf-8")
        _git(self.root, "init", "-q")
        self.checkpoint("Create design")

    def checkpoint(self, message: str) -> bool:
        """Commit the design folder if anything changed. Returns whether a commit was made."""
        if not (self.root / ".git").exists():
            return False
        _git(self.root, "add", "-A")
        if _git(self.root, "diff", "--cached", "--quiet").returncode == 0:
            return False
        run = _git(self.root, "-c", "user.name=3D Print Assist", "-c", "user.email=print-assist@localhost",
                   "commit", "-q", "-m", message)
        return run.returncode == 0

    def history(self, limit: int = 50) -> list[dict]:
        run = _git(self.root, "log", f"-{limit}", "--format=%h%x09%cI%x09%s")
        out = []
        for line in run.stdout.splitlines():
            h, when, msg = line.split("\t", 2)
            out.append({"id": h, "at": when, "message": msg})
        return out

    # ---------- settings ----------

    @property
    def printer(self) -> PrinterProfile:
        all_ = profiles()
        return all_.get(self.data.settings.printer) or all_[DEFAULT_PRINTER]

    # ---------- build ----------

    def _key(self) -> str:
        return self.data.design.model_dump_json() + self.data.rules.model_dump_json()

    def build(self) -> BuildResult:
        """Build the current design, reusing the last result while the design is unchanged."""
        key = self._key()
        if self._build is None or self._build[0] != key:
            self._build = (key, build_design(self.data.design, self.data.rules))
        return self._build[1]

    def validate(self) -> Report:
        key = self._key() + self.data.settings.printer
        if self._report is None or self._report[0] != key:
            build = self.build() if self.data.design.features else None
            self._report = (key, validate(build, self.data.rules, self.printer))
        return self._report[1]

    def summary(self) -> dict:
        return {"slug": self.slug, **json.loads(self.data.model_dump_json())}


class Projects:
    """The designs under a data root."""

    def __init__(self, root: Path | None = None):
        self.root = root or data_root()
        self.root.mkdir(parents=True, exist_ok=True)

    def list(self) -> list[dict]:
        out = []
        for d in sorted(self.root.iterdir()):
            f = d / "project.json"
            if f.exists():
                try:
                    name = json.loads(f.read_text(encoding="utf-8")).get("name", d.name)
                except json.JSONDecodeError:
                    name = d.name
                out.append({"slug": d.name, "name": name, "modified": f.stat().st_mtime,
                            "thumbnail": (d / "thumbnail.png").exists(), "recovery": (d / ".recovery.json").exists()})
        return sorted(out, key=lambda p: -p["modified"])

    def create(self, name: str) -> Project:
        slug = base = slugify(name)
        n = 2
        while (self.root / slug).exists():
            slug, n = f"{base}-{n}", n + 1
        p = Project(self.root / slug)
        p.data.name = name.strip() or "Untitled"
        p.save()
        p.init_git()
        return p

    def save_as(self, src: Project, design, name: str) -> Project:
        """A new design folder holding `design` (the working copy), with src's settings, rules and fonts."""
        import shutil

        p = self.create(name)
        for sub in ("fonts", "references"):
            for f in (src.root / sub).iterdir():
                if f.is_file():
                    shutil.copy2(f, p.root / sub / f.name)
        p.data.design = design.model_copy(deep=True)
        p.data.rules = src.data.rules.model_copy(deep=True)
        p.data.settings = src.data.settings.model_copy(deep=True)
        p.save()
        p.checkpoint(f"Save as, from {src.data.name}")
        return p

    def rename(self, slug: str, name: str) -> Project:
        p = self.open(slug)
        new = slugify(name)
        if new != slug:
            if (self.root / new).exists():
                raise ValueError(f"a design called {name!r} already exists")
            p.root.rename(self.root / new)
            p = Project(self.root / new)
        p.data.name = name.strip()
        p.save()
        p.checkpoint(f"Rename to {p.data.name}")
        return p

    def delete(self, slug: str) -> Path:
        """Move a design's folder to the Trash (nothing is erased)."""
        import shutil
        import time

        p = self.open(slug)
        trash = Path.home() / ".Trash"
        trash.mkdir(exist_ok=True)
        dest = trash / f"{p.root.name} {time.strftime('%Y-%m-%d %H.%M.%S')}"
        shutil.move(str(p.root), dest)
        return dest

    def open(self, slug: str) -> Project:
        path = (self.root / slug).resolve()
        if path.parent != self.root.resolve() or not (path / "project.json").exists():
            raise KeyError(slug)
        return Project(path)
