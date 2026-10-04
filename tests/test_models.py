"""Real models. Slow and needs internet once: pytest -m models"""
import pytest

from closet_vision import cutout, pipeline
from conftest import draw_photo

pytestmark = pytest.mark.models


def test_rembg_cuts_out_the_garment():
    g = cutout.extract(draw_photo(), model="u2netp")
    w, h = g.size
    assert 200 < w < 420 and 250 < h < 460      # roughly the shirt, not the wall


def test_fashion_clip_knows_a_skirt_from_a_tee():
    from closet_vision import load_tagger
    tagger = load_tagger()
    skirt = pipeline.tag(cutout.extract(draw_photo("skirt", (20, 20, 20)), model="u2netp"), tagger)
    tee = pipeline.tag(cutout.extract(draw_photo("tee"), model="u2netp"), tagger)
    print(skirt, tee)
    assert skirt.slot == "bottom"
    assert tee.slot == "top"
