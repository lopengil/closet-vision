"""Hang a cut-out on a drawn hanger (or set shoes/bags on a shelf shadow)."""
from __future__ import annotations

from PIL import Image, ImageDraw, ImageFilter

SS = 4  # supersampling for smooth, anti-aliased lines
WOOD, WOOD_DARK, METAL = (196, 150, 98), (140, 98, 58), (120, 120, 128)


def _hook(d: ImageDraw.ImageDraw, cx: int, top: int, bar_y: int, s: int):
    r = 18 * s
    d.arc([cx - r, top, cx + r, top + 2 * r], 180, 400, fill=METAL, width=5 * s)
    d.line([cx + int(r * 0.6), top + int(r * 1.8), cx, bar_y], fill=METAL, width=5 * s)


def _top_hanger(w: int, s: int) -> Image.Image:
    """Classic wooden shoulder hanger, `w` px wide (before supersampling)."""
    W, H = w * s, 110 * s
    img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    cx = W // 2
    _hook(d, cx, 2 * s, 54 * s, s)
    d.polygon([(4 * s, H - 14 * s), (cx, 50 * s), (W - 4 * s, H - 14 * s),
               (W - 4 * s, H - 2 * s), (4 * s, H - 2 * s)], fill=WOOD, outline=WOOD_DARK)
    d.line([(10 * s, H - 9 * s), (W - 10 * s, H - 9 * s)], fill=WOOD_DARK, width=s)
    return img


def _clip_hanger(w: int, s: int) -> Image.Image:
    """Bar with two clips for skirts/trousers."""
    W, H = w * s, 96 * s
    img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    cx = W // 2
    _hook(d, cx, 2 * s, 54 * s, s)
    d.rounded_rectangle([0, 52 * s, W, 62 * s], radius=4 * s, fill=WOOD, outline=WOOD_DARK)
    return img


def _clips(w: int, s: int, xs=(0.08, 0.92)) -> Image.Image:
    W, H = w * s, 96 * s
    img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    for f in xs:
        x = int(W * f) - 8 * s
        d.rounded_rectangle([x, 58 * s, x + 16 * s, H - 2 * s], radius=3 * s,
                            fill=(60, 60, 66), outline=(30, 30, 34))
    return img


def find_old_clips(g: Image.Image) -> tuple[float, float]:
    """Where the photo's own clips were (dark bits along the waistband), as
    fractions of the width, so ours cover them. Falls back to the edges."""
    import numpy as np
    a = np.asarray(g.convert("RGBA"), dtype=float)
    band = a[: max(4, g.height // 14)]
    lum = band[..., :3].mean(2)
    body = a[g.height // 4: g.height * 3 // 4]
    body_lum = np.median(body[..., :3].mean(2)[body[..., 3] > 128]) if (body[..., 3] > 128).any() else 128
    dark = ((lum < body_lum - 45) & (band[..., 3] > 128)).sum(0)
    cols = np.nonzero(dark >= 2)[0]
    if len(cols) < 2:
        return (0.12, 0.88)
    left, right = cols[cols < g.width / 2], cols[cols >= g.width / 2]
    if not len(left) or not len(right):
        return (0.12, 0.88)
    xs = (float(np.median(left)) / g.width, float(np.median(right)) / g.width)
    # real clips sit well apart, near the sides; anything else is noise
    return xs if xs[1] - xs[0] > 0.4 else (0.12, 0.88)


def _down(img: Image.Image, s: int) -> Image.Image:
    return img.resize((img.width // s, img.height // s), Image.LANCZOS)


def on_hanger(cutout: Image.Image, slot: str, size: tuple[int, int] = (480, 640)) -> Image.Image:
    """Return a transparent `size` image with the garment hanging (or standing)."""
    W, H = size
    canvas = Image.new("RGBA", size, (0, 0, 0, 0))
    g = cutout.copy()
    hang = slot in ("top", "dress", "outer", "bottom")
    room_top = 100 if hang else 0
    g.thumbnail((int(W * 0.86), H - room_top - 30))
    gx = (W - g.width) // 2

    if not hang:  # shoes/accessories sit on a soft shadow
        gy = H - g.height - 30
        shadow = Image.new("RGBA", size, (0, 0, 0, 0))
        ImageDraw.Draw(shadow).ellipse([gx, H - 46, gx + g.width, H - 18], fill=(0, 0, 0, 70))
        canvas.alpha_composite(shadow.filter(ImageFilter.GaussianBlur(6)))
        canvas.alpha_composite(g, (gx, gy))
        return canvas

    gy = room_top - 6
    if slot == "bottom":
        hw = int(g.width * 1.04)
        hx = (W - hw) // 2
        canvas.alpha_composite(_down(_clip_hanger(hw, SS), SS), (hx, gy - 58))
        canvas.alpha_composite(g, (gx, gy))
        fx = [(gx - hx + f * g.width) / hw for f in find_old_clips(g)]
        canvas.alpha_composite(_down(_clips(hw, SS, fx), SS), (hx, gy - 58))
    else:
        hw = int(g.width * 0.62)
        hx = (W - hw) // 2
        canvas.alpha_composite(_down(_top_hanger(hw, SS), SS), (hx, gy - 74))
        canvas.alpha_composite(g, (gx, gy))
    return canvas
