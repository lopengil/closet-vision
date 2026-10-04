"""What clothes can be: labels the tagger chooses from, and sensible defaults.

Each kind maps to a slot (top, bottom, dress, outer, shoes, accessory) plus
default formality (1 gym .. 5 gala), warmth (0 summer .. 3 winter) and whether
it keeps rain out. Edit freely: the tagger only ever picks from these lists.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Kind:
    label: str          # what the model is asked about ("a photo of a <label>")
    kind: str           # short name stored in the closet
    slot: str
    formality: int = 2
    warmth: int = 1
    waterproof: bool = False


KINDS: list[Kind] = [
    # tops
    Kind("t-shirt", "tee", "top", 1, 0),
    Kind("graphic t-shirt", "tee", "top", 1, 0),
    Kind("tank top", "tank", "top", 1, 0),
    Kind("crop top", "crop", "top", 2, 0),
    Kind("blouse", "blouse", "top", 3, 1),
    Kind("button-up shirt", "shirt", "top", 3, 1),
    Kind("polo shirt", "polo", "top", 2, 0),
    Kind("knit sweater", "sweater", "top", 2, 2),
    Kind("turtleneck sweater", "turtleneck", "top", 3, 2),
    Kind("cardigan", "cardigan", "top", 2, 2),
    Kind("hoodie", "hoodie", "top", 1, 2),
    Kind("sweatshirt", "sweatshirt", "top", 1, 2),
    Kind("sports bra", "tank", "top", 1, 0),
    # bottoms
    Kind("jeans", "jeans", "bottom", 2, 1),
    Kind("trousers", "pants", "bottom", 3, 1),
    Kind("cargo pants", "pants", "bottom", 1, 1),
    Kind("hiking pants", "pants", "bottom", 1, 1),
    Kind("leggings", "leggings", "bottom", 1, 1),
    Kind("joggers", "joggers", "bottom", 1, 1),
    Kind("shorts", "shorts", "bottom", 1, 0),
    Kind("mini skirt", "skirt", "bottom", 2, 0),
    Kind("midi skirt", "midi skirt", "bottom", 3, 0),
    Kind("maxi skirt", "maxi skirt", "bottom", 3, 1),
    Kind("pencil skirt", "pencil skirt", "bottom", 4, 0),
    # one-pieces
    Kind("mini dress", "dress", "dress", 3, 0),
    Kind("midi dress", "midi dress", "dress", 3, 1),
    Kind("maxi dress", "maxi dress", "dress", 3, 1),
    Kind("slip dress", "slip dress", "dress", 4, 0),
    Kind("jumpsuit", "jumpsuit", "dress", 3, 1),
    # outerwear
    Kind("blazer", "blazer", "outer", 4, 1),
    Kind("denim jacket", "jacket", "outer", 2, 1),
    Kind("leather jacket", "jacket", "outer", 2, 1),
    Kind("bomber jacket", "jacket", "outer", 2, 1),
    Kind("wool coat", "coat", "outer", 3, 3),
    Kind("trench coat", "trench", "outer", 3, 2, True),
    Kind("rain jacket", "raincoat", "outer", 1, 1, True),
    Kind("puffer jacket", "parka", "outer", 1, 3),
    Kind("fleece jacket", "fleece", "outer", 1, 2),
    # shoes
    Kind("sneakers", "sneakers", "shoes", 1, 0),
    Kind("running shoes", "sneakers", "shoes", 1, 0),
    Kind("hiking boots", "boots", "shoes", 1, 1, True),
    Kind("ankle boots", "boots", "shoes", 3, 1),
    Kind("knee-high boots", "kneehigh boots", "shoes", 3, 1),
    Kind("rain boots", "rainboots", "shoes", 1, 1, True),
    Kind("high heels", "heels", "shoes", 4, 0),
    Kind("sandals", "sandals", "shoes", 2, 0),
    Kind("loafers", "loafers", "shoes", 3, 0),
    Kind("ballet flats", "flats", "shoes", 3, 0),
    # accessories
    Kind("handbag", "bag", "accessory", 3, 0),
    Kind("backpack", "backpack", "accessory", 1, 0),
    Kind("hat", "hat", "accessory", 2, 0),
    Kind("baseball cap", "cap", "accessory", 1, 0),
    Kind("beret", "beret", "accessory", 2, 0),
    Kind("scarf", "scarf", "accessory", 2, 1),
    Kind("belt", "belt", "accessory", 3, 0),
    Kind("necklace", "necklace", "accessory", 3, 0),
    Kind("sunglasses", "sunglasses", "accessory", 2, 0),
    Kind("earrings", "earrings", "accessory", 3, 0),
    Kind("watch", "watch", "accessory", 3, 0),
]

PATTERNS = {  # model label -> stored value
    "solid color": "solid", "plaid": "plaid", "stripes": "stripes",
    "polka dots": "dots", "floral print": "floral", "animal print": "animal",
    "graphic print": "graphic",
}

MATERIALS = ["cotton", "denim", "leather", "wool knit", "silk satin", "linen",
             "sequins", "fleece", "nylon", "lace"]

STYLES = ["casual", "classic", "preppy", "sporty", "elegant", "romantic",
          "party", "grunge", "boho", "outdoor"]

# material/style nudges on top of the kind's defaults
FORMALITY_NUDGE = {"silk satin": 1, "sequins": 1, "lace": 1, "fleece": -1,
                   "elegant": 1, "sporty": -1, "outdoor": -1}
WARMTH_NUDGE = {"wool knit": 1, "fleece": 1, "linen": -1}
WATERPROOF_MATERIALS = {"nylon", "leather"}


def by_label(label: str) -> Kind:
    for k in KINDS:
        if k.label == label:
            return k
    raise KeyError(label)
