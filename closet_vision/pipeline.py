"""photo -> cut-out + hanger picture + suggested tags (with confidence)."""
from __future__ import annotations

import base64
import io
from dataclasses import dataclass, field
from typing import Optional

from PIL import Image

from . import colors, cutout, hanger, taxonomy
from .tagger import Tagger


@dataclass
class Suggestion:
    """Everything the closet needs, ready for a human to approve or correct."""
    name: str
    slot: str
    kind: str
    colors: list[str]
    pattern: str
    material: Optional[str]
    styles: list[str]
    formality: int
    warmth: int
    waterproof: bool
    confidence: dict[str, float] = field(default_factory=dict)  # field -> 0..1
    alternatives: dict[str, list[str]] = field(default_factory=dict)


@dataclass
class Result:
    cutout: Image.Image      # garment only, transparent background
    hanger: Image.Image      # garment on a hanger / shelf, transparent
    suggestion: Suggestion

    def to_json(self) -> dict:
        def png(img):
            buf = io.BytesIO()
            img.save(buf, "PNG")
            return base64.b64encode(buf.getvalue()).decode()
        return {"cutout_png": png(self.cutout), "hanger_png": png(self.hanger),
                "suggestion": self.suggestion.__dict__}


def _clamp(v, lo, hi):
    return max(lo, min(hi, v))


def _on_white(rgba: Image.Image) -> Image.Image:
    bg = Image.new("RGB", rgba.size, (255, 255, 255))
    bg.paste(rgba, (0, 0), rgba)
    return bg


def tag(garment: Image.Image, tagger: Optional[Tagger]) -> Suggestion:
    """Guess what the garment is. Without a tagger you still get colours."""
    found = colors.dominant_colors(garment)
    color_names = [n for n, _ in found] or ["grey"]
    conf = {"colors": found[0][1] if found else 0.0}
    alts: dict[str, list[str]] = {}

    if tagger is None:  # colours only: leave the rest for the human
        return Suggestion(name=f"{color_names[0].capitalize()} piece", slot="", kind="",
                          colors=color_names, pattern="solid", material=None, styles=[],
                          formality=2, warmth=1, waterproof=False,
                          confidence={**conf, "kind": 0.0, "pattern": 0.0})
    else:
        img = _on_white(garment)
        kinds = tagger.rank(img, [k.label for k in taxonomy.KINDS])
        kind = taxonomy.by_label(kinds[0][0])
        alts["kind"] = [l for l, _ in kinds[1:4]]
        pats = tagger.rank(img, list(taxonomy.PATTERNS), f"a {kind.label} with {{}}")
        pattern = taxonomy.PATTERNS[pats[0][0]]
        mats = tagger.rank(img, taxonomy.MATERIALS, f"a {kind.label} made of {{}}")
        material = mats[0][0] if mats[0][1] >= 0.3 else None
        sty = tagger.rank(img, taxonomy.STYLES, "a {} style outfit piece")
        styles = [s for s, p in sty[:2] if p >= 0.15] or [sty[0][0]]
        conf.update(kind=round(kinds[0][1], 2), pattern=round(pats[0][1], 2),
                    material=round(mats[0][1], 2), styles=round(sty[0][1], 2))

    formality = kind.formality + sum(taxonomy.FORMALITY_NUDGE.get(x, 0)
                                     for x in [material, *styles] if x)
    warmth = kind.warmth + taxonomy.WARMTH_NUDGE.get(material or "", 0)
    waterproof = kind.waterproof or (
        kind.slot in ("outer", "shoes") and material in taxonomy.WATERPROOF_MATERIALS)

    words = [color_names[0]]
    if material in ("leather", "denim", "silk satin", "wool knit", "linen", "sequins", "lace"):
        words.append(material.split()[0])
    words.append(kind.label)
    name = " ".join(words).capitalize()
    return Suggestion(name=name, slot=kind.slot, kind=kind.kind, colors=color_names,
                      pattern=pattern, material=material, styles=styles,
                      formality=_clamp(formality, 1, 5), warmth=_clamp(warmth, 0, 3),
                      waterproof=waterproof, confidence=conf, alternatives=alts)


def process(photo, tagger: Optional[Tagger] = None, *,
            model: str = cutout.DEFAULT_MODEL, hanger_in_photo: bool = True) -> Result:
    garment = cutout.extract(photo, model=model, hanger_in_photo=hanger_in_photo)
    suggestion = tag(garment, tagger)
    return Result(cutout=garment, hanger=hanger.on_hanger(garment, suggestion.slot),
                  suggestion=suggestion)
