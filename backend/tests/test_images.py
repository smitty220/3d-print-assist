"""Generated images: made from a prompt through a scripted generator (never the real API), kept with their prompt,
shown in concept drawings."""

import base64

import pytest
from fastapi.testclient import TestClient

from print_assist import app as app_module
from print_assist.images import ImageError, Images, _last_image, inline

PNG = base64.b64decode("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8DwHwAFBQIAX8jx0gAAAABJRU5ErkJggg==")
B64 = base64.b64encode(PNG).decode()


@pytest.fixture(autouse=True)
def key(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")


def scripted(calls):
    def post(url, headers, body):
        calls.append((url, headers, body))
        return {"id": "x", "steps": [{"type": "model_output", "content": [{"type": "text", "text": "Here"},
                                                                          {"type": "image", "mime_type": "image/png", "data": B64}]}]}
    return post


def test_the_last_image_block_is_found_in_any_response_shape():
    assert _last_image({"output_image": {"data": "A", "mime_type": "image/jpeg"}}) == ("A", "image/jpeg")
    assert _last_image({"steps": [{"content": [{"type": "image", "data": "A"}]}, {"type": "image", "data": "B"}]})[0] == "B"
    assert _last_image({"steps": [{"type": "text", "text": "no"}]}) is None


def test_an_image_is_kept_with_its_prompt_and_can_start_the_next(tmp_path):
    calls = []
    imgs = Images(tmp_path / "images")
    a = imgs.generate("A flat two-colour dragon", aspect="4:3", post=scripted(calls))
    b = imgs.generate("Same dragon, wings folded", quality="best", size="2K", from_images=["I1"], post=scripted(calls))
    assert (a.id, a.file, b.id) == ("I1", "I1.png", "I2")
    assert (tmp_path / "images" / "I1.png").read_bytes() == PNG
    assert calls[0][1]["x-goog-api-key"] == "test-key"
    assert calls[0][2]["model"] == "gemini-nano-banana-2.1"
    assert calls[0][2]["response_format"] == {"type": "image", "aspect_ratio": "4:3", "image_size": "1K"}
    # The second call sends the first picture back with its prompt, to the best model, at the 2K price.
    assert calls[1][2]["model"] == "gemini-3-pro-image" and calls[1][2]["input"][1]["data"] == B64
    assert (b.cost_usd, b.from_images) == (0.134, ["I1"])
    again = Images(tmp_path / "images")
    assert [(i.id, i.prompt) for i in again.items] == [("I1", "A flat two-colour dragon"), ("I2", "Same dragon, wings folded")]


def test_bad_requests_are_refused_before_anything_is_paid_for(tmp_path, monkeypatch):
    calls = []
    imgs = Images(tmp_path / "images")
    with pytest.raises(ImageError, match="aspect"):
        imgs.generate("x", aspect="7:1", post=scripted(calls))
    with pytest.raises(ImageError, match="1K"):
        imgs.generate("x", quality="draft", size="2K", post=scripted(calls))
    with pytest.raises(ImageError, match="no image 'I9'"):
        imgs.generate("x", from_images=["I9"], post=scripted(calls))
    monkeypatch.delenv("GEMINI_API_KEY")
    monkeypatch.setattr("print_assist.images.load_env_key", lambda: None)
    with pytest.raises(ImageError, match="GEMINI_API_KEY"):
        imgs.generate("x", post=scripted(calls))
    assert calls == []
    with pytest.raises(ImageError, match="no image back"):
        monkeypatch.setenv("GEMINI_API_KEY", "k")
        imgs.generate("x", post=lambda *a: {"steps": [{"type": "text", "text": "I can't draw that"}]})


def test_concept_drawings_show_the_designs_images(tmp_path):
    imgs = Images(tmp_path / "images")
    imgs.generate("dragon", post=scripted([]))
    svg = '<svg><image href="images/I1.png" width="10" height="10"/><image href="images/I7.png"/></svg>'
    out = inline(svg, imgs)
    assert f'href="data:image/png;base64,{B64}"' in out and 'href="images/I7.png"' in out


def test_the_api_makes_lists_and_serves_images(monkeypatch):
    monkeypatch.setattr("print_assist.images._post", scripted([]))
    c = TestClient(app_module.app)
    c.post("/api/projects", json={"name": "Dragon lid"})
    made = c.post("/api/images", json={"prompt": "dragon", "aspect": "4:3"}, headers={"X-Actor": "claude-code"}).json()
    assert made["id"] == "I1" and c.get("/api/images").json()[0]["prompt"] == "dragon"
    assert c.get("/api/images/I1").content == PNG
    assert c.post("/api/images", json={"prompt": "x", "aspect": "7:1"}).status_code == 422
    svg = '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 10 10"><image href="images/I1.png" width="10" height="10"/></svg>'
    cid = c.post("/api/concepts", json={"title": "Lid art", "svg": svg}, headers={"X-Actor": "claude-code"}).json()["id"]
    assert "data:image/png;base64," in c.get(f"/api/concepts/{cid}").json()["svg"]
    assert c.get(f"/api/concepts/{cid}/picture").headers["content-type"] == "image/png"
