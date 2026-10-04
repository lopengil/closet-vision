"""Style fingerprints and outfit compatibility, from Dressify.

Dressify (huggingface.co/Stylique/dressify-models, MIT) is two small models
trained on Polyvore outfits:

* a ResNet50 that turns one item picture into a 512-number "style fingerprint"
  (L2-normalised, so similar items have a dot product close to 1), and
* a transformer that reads the fingerprints of an outfit and scores how well
  they go together (0..1).

Fingerprints are computed once per item, when it's added. Scoring an outfit
then only needs those numbers, which is cheap enough to do for a shortlist
every morning. The model classes below are copied from the Dressify demo
Space (huggingface.co/spaces/Stylique/recomendation, MIT) so the published
weights load without that repo.

The authors call their reported metrics demo-grade; `closet-vision validate`
checks how well it separates outfits you like from ones you don't.

    pip install "closet-vision[style]"
"""
from __future__ import annotations

import re
import threading
from typing import Iterable, Optional, Sequence

import numpy as np
from PIL import Image

REPO = "Stylique/dressify-models"
ITEM_WEIGHTS = "resnet_item_embedder_best.pth"
OUTFIT_WEIGHTS = "vit_outfit_model_best.pth"
DIM = 512


def _modules():
    """Build the model classes lazily so importing this file needs no torch."""
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torchvision.models as tvm

    class ResNetItemEmbedder(nn.Module):
        def __init__(self, embedding_dim: int = DIM) -> None:
            super().__init__()
            model = tvm.resnet50(weights=None)          # the checkpoint has every weight
            self.backbone = nn.Sequential(*list(model.children())[:-1])
            self.proj = nn.Linear(2048, embedding_dim)

        def forward(self, x):
            return F.normalize(self.proj(self.backbone(x).flatten(1)), p=2, dim=1)

    class OutfitCompatibilityModel(nn.Module):
        def __init__(self, embedding_dim: int = DIM, num_layers: int = 4, num_heads: int = 8,
                     ff_multiplier: int = 4, dropout: float = 0.1) -> None:
            super().__init__()
            layer = nn.TransformerEncoderLayer(
                d_model=embedding_dim, nhead=num_heads, dim_feedforward=ff_multiplier * embedding_dim,
                dropout=dropout, batch_first=True, activation="gelu", norm_first=True)
            self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
            self.compatibility_head = nn.Sequential(
                nn.LayerNorm(embedding_dim), nn.Linear(embedding_dim, embedding_dim // 2),
                nn.GELU(), nn.Linear(embedding_dim // 2, 1))

        def forward(self, tokens):
            return self.compatibility_head(self.encoder(tokens).mean(dim=1)).squeeze(-1)

    return torch, ResNetItemEmbedder, OutfitCompatibilityModel


def _state_dict(path: str) -> dict:
    import torch
    state = torch.load(path, map_location="cpu", weights_only=True)
    for key in ("state_dict", "model_state_dict", "model"):
        if isinstance(state, dict) and isinstance(state.get(key), dict):
            state = state[key]
    # checkpoints saved from DataParallel / torch.compile carry a prefix
    return {re.sub(r"^(module\.|_orig_mod\.)", "", k): v for k, v in state.items()}


def _load(model, state: dict, name: str) -> None:
    """Strict loading: a silently half-loaded model would score at random."""
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"{name} weights don't match the model: missing {missing[:5]}, "
                           f"unexpected {unexpected[:5]}")


def on_white(img: Image.Image) -> Image.Image:
    """Polyvore items are product shots on white; match that."""
    if img.mode in ("RGBA", "LA") or "transparency" in img.info:
        rgba = img.convert("RGBA")
        bg = Image.new("RGB", rgba.size, (255, 255, 255))
        bg.paste(rgba, (0, 0), rgba)
        img = bg
    img = img.convert("RGB")
    side = max(img.size)                          # pad to a square, don't stretch
    sq = Image.new("RGB", (side, side), (255, 255, 255))
    sq.paste(img, ((side - img.width) // 2, (side - img.height) // 2))
    return sq


class StyleModel:
    def __init__(self, item_weights: Optional[str] = None, outfit_weights: Optional[str] = None):
        torch, Embedder, Outfit = _modules()
        if item_weights is None or outfit_weights is None:
            from huggingface_hub import hf_hub_download
            item_weights = item_weights or hf_hub_download(REPO, ITEM_WEIGHTS)
            outfit_weights = outfit_weights or hf_hub_download(REPO, OUTFIT_WEIGHTS)
        self.torch = torch
        self.embedder = Embedder()
        _load(self.embedder, _state_dict(item_weights), "item embedder")
        state = _state_dict(outfit_weights)
        layers = 1 + max(int(m.group(1)) for k in state
                         if (m := re.match(r"encoder\.layers\.(\d+)\.", k)))
        self.outfit = Outfit(num_layers=layers)
        _load(self.outfit, state, "outfit model")
        self.embedder.eval()
        self.outfit.eval()
        self._lock = threading.Lock()

    def _tensor(self, img: Image.Image):
        t = self.torch
        im = on_white(img).resize((224, 224), Image.BICUBIC)
        a = np.asarray(im, dtype=np.float32) / 255
        a = (a - (0.485, 0.456, 0.406)) / (0.229, 0.224, 0.225)
        return t.from_numpy(a.transpose(2, 0, 1).astype(np.float32))

    def embed(self, images: Sequence[Image.Image]) -> np.ndarray:
        """Item pictures -> (n, 512) style fingerprints."""
        if not images:
            return np.zeros((0, DIM), np.float32)
        with self._lock, self.torch.inference_mode():
            batch = self.torch.stack([self._tensor(i) for i in images])
            return self.embedder(batch).numpy()

    def score(self, outfits: Iterable[Sequence[Sequence[float]]]) -> list[float]:
        """Each outfit is a list of fingerprints; returns 0..1 per outfit."""
        out = []
        with self._lock, self.torch.inference_mode():
            for items in outfits:
                if len(items) < 2:
                    out.append(0.5)                 # nothing to compare
                    continue
                x = self.torch.tensor(np.asarray(items, dtype=np.float32)).unsqueeze(0)
                out.append(float(self.torch.sigmoid(self.outfit(x))[0]))
        return out


def similar_pairs(embeddings: dict[str, Sequence[float]], threshold: float = 0.92):
    """Near-duplicates: item id pairs whose fingerprints are this close (cosine)."""
    ids = list(embeddings)
    if len(ids) < 2:
        return []
    m = np.asarray([embeddings[i] for i in ids], dtype=np.float32)
    m /= np.linalg.norm(m, axis=1, keepdims=True) + 1e-8
    sim = m @ m.T
    pairs = [(ids[a], ids[b], float(sim[a, b]))
             for a in range(len(ids)) for b in range(a + 1, len(ids)) if sim[a, b] >= threshold]
    return sorted(pairs, key=lambda p: -p[2])


def auc(pos: Sequence[float], neg: Sequence[float]) -> float:
    """Chance that a liked outfit scores above a disliked one (0.5 = coin flip)."""
    if not pos or not neg:
        return float("nan")
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return wins / (len(pos) * len(neg))
