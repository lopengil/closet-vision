import numpy as np
from PIL import Image

from closet_vision import colors, cutout, hanger


def test_strip_hanger_removes_hook_and_bar(cutout_rgba):
    out = cutout.crop(cutout.strip_hanger(cutout.keep_main_object(cutout_rgba)))
    # the top of what's left is the shirt's shoulders, not a 4px hook
    a = np.asarray(out.getchannel("A")) > 128
    assert a[:6].sum(1).max() > 100
    assert out.height < 400


def test_keep_main_object_drops_specks(cutout_rgba):
    a = np.asarray(cutout_rgba).copy()
    a[600:604, 10:14] = (0, 0, 0, 255)          # a speck
    cleaned = cutout.keep_main_object(Image.fromarray(a))
    assert cleaned.getpixel((12, 602))[3] == 0


def test_dominant_color_names():
    img = Image.new("RGBA", (50, 50), (115, 25, 40, 255))
    assert colors.dominant_colors(img)[0][0] == "burgundy"
    two = Image.new("RGBA", (60, 60), (250, 210, 30, 255))
    two.paste((20, 20, 20, 255), (0, 0, 60, 25))
    names = [n for n, _ in colors.dominant_colors(two)]
    assert names[:2] == ["yellow", "black"]


def test_transparent_pixels_are_ignored():
    img = Image.new("RGBA", (40, 40), (255, 255, 255, 0))
    img.paste((40, 90, 200, 255), (10, 10, 30, 30))
    assert colors.dominant_colors(img)[0][0] == "blue"


def test_hanger_images(cutout_rgba):
    g = cutout.crop(cutout.strip_hanger(cutout_rgba))
    for slot in ("top", "bottom", "shoes"):
        h = hanger.on_hanger(g, slot)
        assert h.size == (480, 640) and h.mode == "RGBA"
        assert h.getchannel("A").getbbox() is not None
