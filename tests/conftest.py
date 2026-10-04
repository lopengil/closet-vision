import numpy as np
import pytest
from PIL import Image, ImageDraw


def draw_photo(kind="tee", color=(200, 30, 40), hanger=True, size=(480, 640)):
    """A fake phone photo: textured wall, a hanger, one garment."""
    rng = np.random.default_rng(1)
    wall = (rng.normal(215, 6, (size[1], size[0], 3))).clip(0, 255).astype("uint8")
    img = Image.fromarray(wall)
    d = ImageDraw.Draw(img)
    W, H = size
    if hanger:
        d.arc([W / 2 - 18, 60, W / 2 + 18, 96], 180, 360, fill=(60, 60, 60), width=4)
        d.line([W / 2, 78, W / 2, 130], fill=(60, 60, 60), width=4)
        d.line([110, 150, W / 2, 125, W - 110, 150], fill=(60, 60, 60), width=4)
    if kind == "tee":
        d.polygon([(150, 140), (330, 140), (420, 210), (380, 250), (340, 225), (340, 520),
                   (140, 520), (140, 225), (100, 250), (60, 210)], fill=color)
    elif kind == "skirt":
        d.polygon([(150, 150), (330, 150), (390, 440), (90, 440)], fill=color)
    return img


@pytest.fixture
def photo():
    return draw_photo()


@pytest.fixture
def cutout_rgba():
    """What rembg would give back for draw_photo(): garment + hanger opaque."""
    img = draw_photo().convert("RGBA")
    a = np.asarray(img).copy()
    wall = np.abs(a[..., :3].astype(int) - 215).sum(2) < 40
    a[..., 3] = np.where(wall, 0, 255)
    return Image.fromarray(a)


class FakeTagger:
    def __init__(self, picks):
        self.picks = set(picks)

    def rank(self, image, labels, template="a photo of {}"):
        scored = [(l, 0.8 if l in self.picks else 0.2 / len(labels)) for l in labels]
        return sorted(scored, key=lambda t: -t[1])
