"""Images: pictures generated from a prompt (artwork for an inlay or a relief, a creature to model from), kept in the
design's images/ folder beside a JSON record of how each was made, so any of them can be made again or varied.

The generator is Google's Gemini image models (GEMINI_API_KEY from the shell or backend/.env, billed per image).
An image can start from earlier ones ("the same dragon, tail curled the other way"): they go in with the prompt.
A concept drawing shows one with <image href="images/I3.png">; `inline()` turns that into the picture itself.
"""

from __future__ import annotations

import base64
import os
import re
import time
from pathlib import Path
from typing import Any, Callable, Literal

from pydantic import BaseModel, Field

from .ai.client import load_env_key

API = "https://generativelanguage.googleapis.com/v1beta/interactions"
# Quality → model, and the US$ per image Google lists for it (1K / 2K / 4K), to show what a round costs.
MODELS = {
    "draft": ("gemini-3.1-flash-lite-image", {"1K": 0.034}),
    "standard": ("gemini-nano-banana-2.1", {"1K": 0.034, "2K": 0.050, "4K": 0.113}),
    "best": ("gemini-3-pro-image", {"1K": 0.134, "2K": 0.134, "4K": 0.24}),
}
ASPECTS = ("1:1", "3:2", "2:3", "3:4", "4:3", "4:5", "5:4", "9:16", "16:9", "21:9")

Quality = Literal["draft", "standard", "best"]
Size = Literal["1K", "2K", "4K"]


class ImageError(ValueError):
    pass


class GeneratedImage(BaseModel):
    id: str  # "I1", "I2", ...
    file: str  # "I1.png", in the images folder
    prompt: str
    quality: Quality = "standard"
    model: str = ""
    aspect: str = "1:1"
    size: Size = "1K"
    from_images: list[str] = Field(default_factory=list)  # earlier images it started from
    actor: str = "claude-code"
    cost_usd: float = 0.0  # Google's list price, not a bill
    created: float = Field(default_factory=time.time)


Post = Callable[[str, dict, dict], dict]  # (url, headers, json body) -> response JSON


def _post(url: str, headers: dict, body: dict) -> dict:
    import httpx

    r = httpx.post(url, headers=headers, json=body, timeout=180)
    if r.status_code >= 400:
        try:
            msg = r.json().get("error", {}).get("message", r.text)
        except ValueError:
            msg = r.text
        raise ImageError(f"Gemini refused the request ({r.status_code}): {msg}")
    return r.json()


def _last_image(obj: Any) -> tuple[str, str] | None:
    """The last image block (base64 data, mime type) anywhere in a response: the convenience field `output_image`
    if present, else the last {"type": "image", "data": ...} in its steps."""
    if isinstance(obj, dict):
        if isinstance(obj.get("output_image"), dict) and obj["output_image"].get("data"):
            o = obj["output_image"]
            return o["data"], o.get("mime_type", "image/png")
        found = None
        for v in obj.values():
            found = _last_image(v) or found
        if obj.get("type") == "image" and isinstance(obj.get("data"), str):
            found = (obj["data"], obj.get("mime_type", "image/png"))
        return found
    if isinstance(obj, list):
        found = None
        for v in obj:
            found = _last_image(v) or found
        return found
    return None


EXT = {"image/png": ".png", "image/jpeg": ".jpg", "image/webp": ".webp"}


class Images:
    """The generated images of one design, oldest first."""

    def __init__(self, folder: Path):
        self.folder = folder
        self.items: list[GeneratedImage] = []
        if folder.is_dir():
            for p in sorted(folder.glob("I*.json"), key=lambda p: int(p.stem[1:]) if p.stem[1:].isdigit() else 0):
                try:
                    self.items.append(GeneratedImage.model_validate_json(p.read_text(encoding="utf-8")))
                except ValueError:
                    continue

    def get(self, iid: str) -> GeneratedImage | None:
        return next((i for i in self.items if i.id == iid), None)

    def path(self, iid: str) -> Path:
        img = self.get(iid)
        if img is None:
            raise ImageError(f"no image {iid!r}")
        return self.folder / img.file

    def generate(self, prompt: str, quality: Quality = "standard", aspect: str = "1:1", size: Size = "1K",
                 from_images: list[str] | None = None, actor: str = "claude-code", post: Post | None = None) -> GeneratedImage:
        if not prompt.strip():
            raise ImageError("an image needs a prompt")
        if quality not in MODELS:
            raise ImageError(f"quality must be one of {', '.join(MODELS)}")
        model, prices = MODELS[quality]
        if aspect not in ASPECTS:
            raise ImageError(f"aspect must be one of {', '.join(ASPECTS)}")
        if size not in prices:
            raise ImageError(f"{quality} images come in {', '.join(prices)}")
        inputs: list[dict] = [{"type": "text", "text": prompt.strip()}]
        for iid in from_images or []:
            p = self.path(iid)
            mime = next((m for m, e in EXT.items() if e == p.suffix), "image/png")
            inputs.append({"type": "image", "mime_type": mime, "data": base64.b64encode(p.read_bytes()).decode()})
        load_env_key()
        key = os.environ.get("GEMINI_API_KEY")
        if not key:
            raise ImageError("No Gemini API key: set GEMINI_API_KEY in your shell or in backend/.env")
        body = {"model": model, "input": inputs,
                "response_format": {"type": "image", "aspect_ratio": aspect, "image_size": size}}
        found = _last_image((post or _post)(API, {"x-goog-api-key": key, "Content-Type": "application/json"}, body))
        if found is None:
            raise ImageError("Gemini sent no image back (it may have declined the prompt); try rewording it")
        data, mime = found
        n = max((int(i.id[1:]) for i in self.items if i.id[1:].isdigit()), default=0) + 1
        img = GeneratedImage(id=f"I{n}", file=f"I{n}{EXT.get(mime, '.png')}", prompt=prompt.strip(), quality=quality,
                             model=model, aspect=aspect, size=size, from_images=list(from_images or []), actor=actor,
                             cost_usd=prices[size])
        self.folder.mkdir(parents=True, exist_ok=True)
        (self.folder / img.file).write_bytes(base64.b64decode(data))
        (self.folder / f"{img.id}.json").write_text(img.model_dump_json(indent=1), encoding="utf-8")
        self.items.append(img)
        return img

    def add_upload(self, data: bytes, ext: str, note: str = "", actor: str = "user") -> GeneratedImage:
        if not data:
            raise ImageError("uploaded image is empty")
        norm_ext = ext.lower()
        if norm_ext == ".jpeg":
            norm_ext = ".jpg"
        if norm_ext in (".heic", ".heif"):
            import io
            import pillow_heif
            from PIL import Image, ImageOps

            pillow_heif.register_heif_opener()
            try:
                with Image.open(io.BytesIO(data)) as im:
                    im = ImageOps.exif_transpose(im).convert("RGB")
                    buf = io.BytesIO()
                    im.save(buf, format="JPEG", quality=92)
                    data = buf.getvalue()
            except Exception as exc:
                raise ImageError(f"could not decode HEIC/HEIF image: {exc}") from exc
            norm_ext = ".jpg"
        if norm_ext not in (".png", ".jpg", ".webp"):
            raise ImageError("only PNG, JPEG, WebP, and HEIC/HEIF images are supported")
        n = max((int(i.id[1:]) for i in self.items if i.id[1:].isdigit()), default=0) + 1
        img = GeneratedImage(
            id=f"I{n}",
            file=f"I{n}{norm_ext}",
            prompt=(note or "Uploaded reference photo").strip(),
            quality="standard",
            model="upload",
            actor=actor,
            cost_usd=0.0,
        )
        self.folder.mkdir(parents=True, exist_ok=True)
        (self.folder / img.file).write_bytes(data)
        (self.folder / f"{img.id}.json").write_text(img.model_dump_json(indent=1), encoding="utf-8")
        self.items.append(img)
        return img

    def delete(self, iid: str) -> None:
        img = self.get(iid)
        if img is None:
            raise ImageError(f"no image {iid!r}")
        (self.folder / img.file).unlink(missing_ok=True)
        (self.folder / f"{img.id}.json").unlink(missing_ok=True)
        self.items = [i for i in self.items if i.id != iid]


_HREF = re.compile(r"""((?:xlink:)?href\s*=\s*)(["'])images/(I\d+)\.(?:png|jpg|webp)\2""")


def inline(svg: str, images: Images) -> str:
    """The drawing with each <image href="images/I3.png"> replaced by the picture itself (a data: URI), so it shows
    wherever the drawing goes; an id the design does not have is left as it is."""
    def put(m: re.Match) -> str:
        img = images.get(m[3])
        p = images.folder / img.file if img else None
        if p is None or not p.is_file():
            return m[0]
        mime = next((k for k, e in EXT.items() if e == p.suffix), "image/png")
        return f'{m[1]}{m[2]}data:{mime};base64,{base64.b64encode(p.read_bytes()).decode()}{m[2]}'

    return _HREF.sub(put, svg)
