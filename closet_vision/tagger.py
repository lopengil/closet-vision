"""Zero-shot clothing tagger.

Default model: FashionCLIP (patrickjohncyh/fashion-clip), a CLIP model tuned on
fashion product photos. Any CLIP checkpoint on Hugging Face works
(e.g. openai/clip-vit-base-patch32). No training needed: we ask the model which
of our text labels best matches the photo.
"""
from __future__ import annotations

from typing import Protocol, Sequence

from PIL import Image

DEFAULT_MODEL = "patrickjohncyh/fashion-clip"


class Tagger(Protocol):
    def rank(self, image: Image.Image, labels: Sequence[str],
             template: str = "a photo of {}") -> list[tuple[str, float]]:
        """Labels sorted best-first with probabilities that sum to 1."""
        ...


class ClipTagger:
    def __init__(self, model_id: str = DEFAULT_MODEL, device: str = "cpu"):
        import torch
        from transformers import CLIPModel, CLIPProcessor
        self.torch = torch
        self.model = CLIPModel.from_pretrained(model_id).to(device).eval()
        self.proc = CLIPProcessor.from_pretrained(model_id)
        self.device = device
        self._text_cache: dict[tuple, object] = {}

    def _text(self, prompts: tuple[str, ...]):
        if prompts not in self._text_cache:
            self._text_cache[prompts] = self.proc(
                text=list(prompts), return_tensors="pt", padding=True).to(self.device)
        return self._text_cache[prompts]

    def _image(self, image: Image.Image):
        key = id(image)
        if getattr(self, "_img_key", None) != key:
            self._img_key = key
            self._img = self.proc(images=image, return_tensors="pt").to(self.device)
        return self._img

    def rank(self, image, labels, template="a photo of {}"):
        prompts = tuple(template.format(l) for l in labels)
        # the full forward pass returns logits on every transformers version
        # (get_*_features changed return types between releases)
        with self.torch.no_grad():
            out = self.model(**self._text(prompts), **self._image(image))
        probs = out.logits_per_image.softmax(dim=-1)[0].tolist()
        return sorted(zip(labels, probs), key=lambda t: -t[1])
