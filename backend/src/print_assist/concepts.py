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
    mode: Literal["engineer", "character"] = "engineer"
    image_file: str = ""  # Character Mode turnaround PNG kept in concepts/ (never in Photos/images)
    clean_image_file: str = ""  # Un-sliced source painting used for 3D sculpting when image_file shows flexi cuts
    style: str = "auto"
    flexi_segments: int = 0
    flexi_cuts_pct: list[float] = Field(default_factory=list)
    reference_photos: list[str] = Field(default_factory=list)
    views: list[str] = Field(default_factory=list)  # e.g. ["front", "side", "back"]
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
        self.sync()

    def sync(self) -> None:
        """Merge any concept files added or updated on disk (e.g. by another process)."""
        if not self.folder.is_dir():
            return
        by_id = {c.id: c for c in self.items}
        changed = False
        for p in sorted(self.folder.glob("K*.json"), key=lambda p: int(p.stem[1:]) if p.stem[1:].isdigit() else 0):
            try:
                disk = Concept.model_validate_json(p.read_text(encoding="utf-8"))
            except ValueError:
                continue
            existing = by_id.get(disk.id)
            if existing is None:
                by_id[disk.id] = disk
                changed = True
            elif existing.status != "pending" and disk != existing:
                by_id[disk.id] = disk
                changed = True
        if changed:
            self.items = sorted(
                by_id.values(),
                key=lambda c: int(c.id[1:]) if c.id.startswith("K") and c.id[1:].isdigit() else 0,
            )

    def _save(self, c: Concept) -> None:
        self.folder.mkdir(parents=True, exist_ok=True)
        (self.folder / f"{c.id}.json").write_text(c.model_dump_json(indent=1), encoding="utf-8")

    def get(self, cid: str) -> Concept | None:
        return next((c for c in self.items if c.id == cid), None)

    def pending(self) -> Concept | None:
        return next((c for c in reversed(self.items) if c.status == "pending"), None)

    def dismiss_pending(self, note: str = "cancelled") -> None:
        for c in self.items:
            if c.status == "pending":
                c.status, c.note, c.answered = "revise", note, time.time()
                self._save(c)

    def image_path(self, cid: str) -> Path | None:
        c = self.get(cid)
        if c is None or not c.image_file:
            return None
        p = self.folder / c.image_file
        return p if p.is_file() else None

    def clean_image_path(self, cid: str) -> Path | None:
        c = self.get(cid)
        if c is None:
            return None
        if c.clean_image_file:
            p_clean = self.folder / c.clean_image_file
            if p_clean.is_file():
                return p_clean
        alt_clean = self.folder / f"{c.id}_clean.png"
        if alt_clean.is_file():
            return alt_clean
        return self.image_path(cid)

    def propose(self, title: str, svg: str = "", summary: str = "", assumptions: list[str] | None = None,
                questions: list[str] | None = None, actor: str = "claude-code",
                mode: Literal["engineer", "character"] = "engineer",
                image_bytes: bytes | None = None, image_ext: str = ".png",
                reference_photos: list[str] | None = None,
                views: list[str] | None = None,
                clean_image_bytes: bytes | None = None,
                style: str = "auto",
                flexi_segments: int = 0,
                flexi_cuts_pct: list[float] | None = None) -> Concept:
        if not title.strip():
            raise ConceptError("a concept needs a title")
        n = max((int(c.id[1:]) for c in self.items if c.id[1:].isdigit()), default=0) + 1
        cid = f"K{n}"
        image_file = ""
        clean_image_file = ""
        if image_bytes is not None:
            import base64
            import io
            from PIL import Image as PILImage

            self.folder.mkdir(parents=True, exist_ok=True)
            ext = image_ext if image_ext in (".png", ".jpg", ".webp") else ".png"
            image_file = f"{cid}{ext}"
            (self.folder / image_file).write_bytes(image_bytes)
            if clean_image_bytes is not None:
                clean_image_file = f"{cid}_clean{ext}"
                (self.folder / clean_image_file).write_bytes(clean_image_bytes)
            with PILImage.open(io.BytesIO(image_bytes)) as im:
                iw, ih = im.size
            mime = "image/jpeg" if ext == ".jpg" else "image/webp" if ext == ".webp" else "image/png"
            b64 = base64.b64encode(image_bytes).decode()
            drawing = (
                f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {iw} {ih}" width="{iw}" height="{ih}">'
                f'<image href="data:{mime};base64,{b64}" width="{iw}" height="{ih}"/></svg>'
            )
        else:
            drawing = check_svg(svg)
        for old in self.items:  # a new concept replaces one still waiting
            if old.status == "pending":
                old.status, old.note, old.answered = "revise", "replaced by a newer concept", time.time()
                self._save(old)
        c = Concept(
            id=cid, title=title.strip(), summary=summary.strip(), svg=drawing,
            mode=mode, image_file=image_file, clean_image_file=clean_image_file,
            style=style, flexi_segments=max(0, int(flexi_segments)),
            flexi_cuts_pct=list(flexi_cuts_pct or []),
            reference_photos=list(reference_photos or []),
            views=list(views or []),
            assumptions=[a for a in (assumptions or []) if a.strip()],
            questions=[q for q in (questions or []) if q.strip()], actor=actor,
        )
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
