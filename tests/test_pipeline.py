import io

from closet_vision import cutout, pipeline
from conftest import FakeTagger


def test_tag_with_tagger(cutout_rgba):
    g = cutout.crop(cutout.strip_hanger(cutout_rgba))
    s = pipeline.tag(g, FakeTagger({"mini skirt", "solid color", "leather", "party"}))
    assert (s.slot, s.kind, s.pattern, s.material) == ("bottom", "skirt", "solid", "leather")
    assert s.colors[0] == "red"
    assert s.name == "Red leather mini skirt"
    assert s.confidence["kind"] == 0.8


def test_nudges_and_waterproof(cutout_rgba):
    s = pipeline.tag(cutout_rgba, FakeTagger({"rain jacket", "solid color", "nylon", "outdoor"}))
    assert s.slot == "outer" and s.waterproof
    s = pipeline.tag(cutout_rgba, FakeTagger({"knit sweater", "solid color", "wool knit", "casual"}))
    assert s.warmth == 3


def test_without_tagger_still_gives_colors(cutout_rgba):
    s = pipeline.tag(cutout_rgba, None)
    assert s.colors and s.confidence["kind"] == 0.0


def test_server(monkeypatch, cutout_rgba, photo):
    monkeypatch.setenv("CLOSET_VISION_TAGGER", "off")
    monkeypatch.setattr(cutout, "remove_background", lambda img, model=None: cutout_rgba)
    from closet_vision.server import create_app
    buf = io.BytesIO()
    photo.save(buf, "JPEG")
    buf.seek(0)
    r = create_app().test_client().post("/process", data={"photo": (buf, "tee.jpg")})
    body = r.get_json()
    assert r.status_code == 200 and body["hanger_png"] and body["suggestion"]["colors"]
