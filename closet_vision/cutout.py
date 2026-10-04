"""Background removal + clean-up: photo in, garment-only RGBA out."""
from __future__ import annotations

from collections import deque
from functools import lru_cache

import numpy as np
from PIL import Image, ImageFilter, ImageOps

DEFAULT_MODEL = "isnet-general-use"   # best quality; "u2netp" is tiny and fast (Pi)
MAX_SIDE = 1024


@lru_cache(maxsize=2)
def _session(model: str):
    from rembg import new_session  # imported lazily: heavy, and optional in tests
    return new_session(model)


def load(photo) -> Image.Image:
    """Open a path/file/PIL image, fix phone rotation, cap the size."""
    img = photo if isinstance(photo, Image.Image) else Image.open(photo)
    img = ImageOps.exif_transpose(img).convert("RGB")
    img.thumbnail((MAX_SIDE, MAX_SIDE))
    return img


def remove_background(img: Image.Image, model: str = DEFAULT_MODEL) -> Image.Image:
    from rembg import remove
    return remove(img, session=_session(model)).convert("RGBA")


def _components(mask: np.ndarray) -> list[np.ndarray]:
    """4-connected components of a small boolean mask, biggest first."""
    seen = np.zeros_like(mask, dtype=bool)
    h, w = mask.shape
    comps = []
    for y, x in zip(*np.nonzero(mask)):
        if seen[y, x]:
            continue
        pts, q = [], deque([(y, x)])
        seen[y, x] = True
        while q:
            cy, cx = q.popleft()
            pts.append((cy, cx))
            for ny, nx in ((cy + 1, cx), (cy - 1, cx), (cy, cx + 1), (cy, cx - 1)):
                if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and not seen[ny, nx]:
                    seen[ny, nx] = True
                    q.append((ny, nx))
        comps.append(np.array(pts))
    return sorted(comps, key=len, reverse=True)


def keep_main_object(rgba: Image.Image, min_ratio: float = 0.08) -> Image.Image:
    """Drop specks and stray background blobs; keep the garment (and a pair's
    second shoe, which is why big secondary pieces survive)."""
    a = np.asarray(rgba.getchannel("A"))
    scale = 192 / max(a.shape)
    small = np.asarray(rgba.getchannel("A").resize(
        (max(1, int(a.shape[1] * scale)), max(1, int(a.shape[0] * scale))))) > 128
    comps = _components(small)
    if not comps:
        return rgba
    keep = np.zeros_like(small)
    for c in comps:
        if len(c) >= len(comps[0]) * min_ratio:
            keep[c[:, 0], c[:, 1]] = True
    keep_img = Image.fromarray(keep.astype(np.uint8) * 255).resize(
        rgba.size, Image.NEAREST).filter(ImageFilter.MaxFilter(5))
    out = rgba.copy()
    out.putalpha(Image.fromarray(np.minimum(a, np.asarray(keep_img))))
    return out


def strip_hanger(rgba: Image.Image) -> Image.Image:
    """Cut off a hanger hook/clips sticking out above the garment.

    Garment rows are wide; hook and clip rows are narrow. Everything above the
    first run of 'wide' rows goes.
    """
    a = np.asarray(rgba.getchannel("A")) > 128
    width = a.sum(1)
    if width.max() == 0:
        return rgba
    wide = width >= 0.45 * np.percentile(width[width > 0], 90)
    run = max(3, a.shape[0] // 60)
    top = 0
    for y in range(len(wide) - run):
        if wide[y:y + run].all():
            top = y
            break
    out = np.asarray(rgba).copy()
    out[:top, :, 3] = 0
    # the hanger's wire/bar can poke out past the shoulders: an "opening"
    # (erode then dilate) on the top quarter deletes lines thinner than k px
    k = max(3, (a.shape[0] // 80) | 1)
    band = slice(0, top + a.shape[0] // 4)
    alpha = Image.fromarray(out[band, :, 3])
    opened = alpha.filter(ImageFilter.MinFilter(k)).filter(ImageFilter.MaxFilter(k + 2))
    out[band, :, 3] = np.minimum(out[band, :, 3], np.asarray(opened))
    return Image.fromarray(out)


def crop(rgba: Image.Image, pad: int = 4) -> Image.Image:
    box = rgba.getchannel("A").point(lambda v: 255 if v > 24 else 0).getbbox()
    if not box:
        return rgba
    x0, y0, x1, y1 = box
    return rgba.crop((max(0, x0 - pad), max(0, y0 - pad),
                      min(rgba.width, x1 + pad), min(rgba.height, y1 + pad)))


def extract(photo, model: str = DEFAULT_MODEL, hanger_in_photo: bool = True) -> Image.Image:
    rgba = remove_background(load(photo), model)
    rgba = keep_main_object(rgba)
    if hanger_in_photo:
        rgba = strip_hanger(rgba)
    return crop(rgba)
