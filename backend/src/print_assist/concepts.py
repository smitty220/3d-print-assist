"""Concepts: before the AI builds anything for a request, it shows what it understood, as a 2D drawing it makes for
that request (an SVG) with a short summary, its assumptions and its questions, and waits for the user to approve it or
ask for changes. Nothing about a concept is specific to one kind of part.

Concepts are kept in the design's concepts/ folder (one JSON file each), so their history survives a restart.
"""

from __future__ import annotations

import re
import time
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

MAX_SVG = 2_000_000  # bytes

Status = Literal["pending", "approved", "revise"]


class Concept(BaseModel):
    id: str  # "K1", "K2", ...
    title: str
    summary: str = ""
    svg: str = ""
    assumptions: list[str] = Field(default_factory=list)
    questions: list[str] = Field(default_factory=list)
    actor: str = "claude-code"
    status: Status = "pending"
    note: str = ""  # the user's answer: what to change, or anything to keep in mind
    created: float = Field(default_factory=time.time)
    answered: float | None = None


class ConceptError(ValueError):
    pass


def check_svg(svg: str) -> str:
    """The drawing, if it is an SVG document of a sensible size. The app shows it as an image, so scripts in it never
    run; this only catches mistakes early."""
    svg = svg.strip()
    if len(svg.encode()) > MAX_SVG:
        raise ConceptError(f"the drawing is over {MAX_SVG // 1_000_000} MB")
    body = re.sub(r"^<\?xml[^>]*>\s*", "", svg)
    body = re.sub(r"^<!--.*?-->\s*", "", body, flags=re.S)
    if not body.startswith("<svg") or "</svg>" not in body[-200:]:
        raise ConceptError("the drawing must be one <svg>...</svg> document")
    if "xmlns=" not in body.split(">", 1)[0]:
        body = body.replace("<svg", '<svg xmlns="http://www.w3.org/2000/svg"', 1)
    return body


def picture(svg: str, width: int = 1200) -> bytes:
    """The drawing as a PNG `width` pixels wide, so the AI can look at what it drew before (or after) proposing it.
    Links to anything but the drawing itself or inline data are dropped first: a drawing never reads other files."""
    import resvg_py

    body = check_svg(svg)
    body = re.sub(r"""(?:xlink:)?href\s*=\s*(["'])(?!#|data:)[^"']*\1""", "", body)
    width = max(200, min(int(width), 3000))
    return bytes(resvg_py.svg_to_bytes(svg_string=body, width=width))


class Concepts:
    """The concepts of one design, newest last."""

    def __init__(self, folder: Path):
        self.folder = folder
        self.items: list[Concept] = []
        if folder.is_dir():
            for p in sorted(folder.glob("K*.json"), key=lambda p: int(p.stem[1:]) if p.stem[1:].isdigit() else 0):
                try:
                    self.items.append(Concept.model_validate_json(p.read_text(encoding="utf-8")))
                except ValueError:
                    continue

    def _save(self, c: Concept) -> None:
        self.folder.mkdir(parents=True, exist_ok=True)
        (self.folder / f"{c.id}.json").write_text(c.model_dump_json(indent=1), encoding="utf-8")

    def get(self, cid: str) -> Concept | None:
        return next((c for c in self.items if c.id == cid), None)

    def pending(self) -> Concept | None:
        return next((c for c in reversed(self.items) if c.status == "pending"), None)

    def propose(self, title: str, svg: str, summary: str = "", assumptions: list[str] | None = None,
                questions: list[str] | None = None, actor: str = "claude-code") -> Concept:
        if not title.strip():
            raise ConceptError("a concept needs a title")
        drawing = check_svg(svg)
        for old in self.items:  # a new concept replaces one still waiting
            if old.status == "pending":
                old.status, old.note, old.answered = "revise", "replaced by a newer concept", time.time()
                self._save(old)
        n = max((int(c.id[1:]) for c in self.items if c.id[1:].isdigit()), default=0) + 1
        c = Concept(id=f"K{n}", title=title.strip(), summary=summary.strip(), svg=drawing,
                    assumptions=[a for a in (assumptions or []) if a.strip()],
                    questions=[q for q in (questions or []) if q.strip()], actor=actor)
        self.items.append(c)
        self._save(c)
        return c

    def respond(self, cid: str, decision: Literal["approve", "revise"], note: str = "") -> Concept:
        c = self.get(cid)
        if c is None:
            raise ConceptError(f"no concept {cid!r}")
        if c.status != "pending":
            raise ConceptError(f"{cid} was already answered ({c.status})")
        c.status = "approved" if decision == "approve" else "revise"
        c.note, c.answered = note.strip(), time.time()
        self._save(c)
        return c
