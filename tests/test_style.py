"""No torch needed: the maths around Dressify."""
import math

from PIL import Image

from closet_vision.style import auc, on_white, similar_pairs


def test_auc():
    assert auc([0.9, 0.8], [0.1, 0.2]) == 1.0
    assert auc([0.1], [0.9]) == 0.0
    assert auc([0.5], [0.5]) == 0.5
    assert math.isnan(auc([], [0.3]))


def test_similar_pairs_finds_near_duplicates():
    emb = {"black-tee": [1, 0, 0], "black-tee-2": [0.99, 0.05, 0], "red-skirt": [0, 1, 0]}
    pairs = similar_pairs(emb, threshold=0.95)
    assert [(a, b) for a, b, _ in pairs] == [("black-tee", "black-tee-2")]


def test_cutouts_go_on_white_squares():
    cut = Image.new("RGBA", (40, 100), (0, 0, 0, 0))
    out = on_white(cut)
    assert out.size == (100, 100) and out.getpixel((5, 5)) == (255, 255, 255)
