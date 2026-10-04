"""Dominant colours of a cut-out garment, as friendly names."""
from __future__ import annotations

import numpy as np
from PIL import Image

NAMED = {
    "black": (20, 20, 20), "white": (245, 245, 242), "grey": (140, 140, 140),
    "charcoal": (60, 62, 66), "navy": (25, 35, 80), "blue": (40, 90, 200),
    "lightblue": (150, 190, 230), "denim": (70, 100, 150), "teal": (0, 120, 120),
    "green": (40, 130, 60), "olive": (105, 105, 45), "mint": (165, 225, 190),
    "yellow": (250, 210, 30), "mustard": (200, 160, 30), "orange": (240, 130, 30),
    "red": (200, 25, 30), "burgundy": (115, 25, 40), "pink": (240, 140, 180),
    "hotpink": (235, 45, 140), "lilac": (190, 165, 225), "purple": (110, 50, 150),
    "brown": (100, 65, 40), "camel": (190, 140, 85), "tan": (205, 170, 125),
    "beige": (225, 205, 175), "cream": (243, 235, 212), "khaki": (185, 175, 125),
    "silver": (192, 192, 200), "gold": (205, 165, 50),
}


def _lab(rgb: np.ndarray) -> np.ndarray:
    """sRGB (0-255, Nx3) -> CIE Lab, so 'close' means close to the eye."""
    c = rgb / 255.0
    c = np.where(c > 0.04045, ((c + 0.055) / 1.055) ** 2.4, c / 12.92)
    xyz = c @ np.array([[0.4124, 0.2126, 0.0193],
                        [0.3576, 0.7152, 0.1192],
                        [0.1805, 0.0722, 0.9505]])
    xyz /= np.array([0.95047, 1.0, 1.08883])
    f = np.where(xyz > 0.008856, np.cbrt(xyz), 7.787 * xyz + 16 / 116)
    return np.stack([116 * f[:, 1] - 16, 500 * (f[:, 0] - f[:, 1]),
                     200 * (f[:, 1] - f[:, 2])], axis=1)


_NAMES = list(NAMED)
_PALETTE_LAB = _lab(np.array([NAMED[n] for n in _NAMES], dtype=float))


def nearest_name(rgb) -> str:
    d = np.linalg.norm(_PALETTE_LAB - _lab(np.array([rgb], dtype=float)), axis=1)
    return _NAMES[int(d.argmin())]


def _kmeans(x: np.ndarray, k: int, iters: int = 12, seed: int = 0):
    rng = np.random.default_rng(seed)
    centers = [x[rng.integers(len(x))]]
    for _ in range(1, k):  # k-means++ start
        d = np.min([((x - c) ** 2).sum(1) for c in centers], axis=0)
        centers.append(x[rng.choice(len(x), p=d / d.sum())] if d.sum() else x[0])
    centers = np.array(centers)
    for _ in range(iters):
        labels = ((x[:, None, :] - centers[None]) ** 2).sum(2).argmin(1)
        for i in range(k):
            if (labels == i).any():
                centers[i] = x[labels == i].mean(0)
    return centers, labels


def dominant_colors(img: Image.Image, k: int = 5, min_share: float = 0.12,
                    max_colors: int = 3) -> list[tuple[str, float]]:
    """[(name, share)] for the opaque pixels, main colour first."""
    rgba = np.asarray(img.convert("RGBA"), dtype=float).reshape(-1, 4)
    px = rgba[rgba[:, 3] > 200][:, :3]
    if len(px) == 0:
        return []
    if len(px) > 6000:
        px = px[np.random.default_rng(0).choice(len(px), 6000, replace=False)]
    k = min(k, len(np.unique(px, axis=0)))
    centers, labels = _kmeans(_lab(px), k)
    shares: dict[str, float] = {}
    for i in range(k):
        share = float((labels == i).mean())
        if share == 0:
            continue
        rgb = px[labels == i].mean(0)
        name = nearest_name(rgb)
        shares[name] = shares.get(name, 0) + share
    ranked = sorted(shares.items(), key=lambda t: -t[1])
    out = [(n, round(s, 2)) for n, s in ranked if s >= min_share][:max_colors]
    return out or [(ranked[0][0], round(ranked[0][1], 2))]
