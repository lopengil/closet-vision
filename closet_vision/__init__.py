"""closet-vision: turn a photo of a clothing item into a clean cut-out on a
hanger plus suggested tags (slot, kind, colours, pattern, material, style).

    from closet_vision import process, load_tagger
    result = process("skirt.jpg", load_tagger())
    result.hanger.save("skirt_on_hanger.png")
    print(result.suggestion)
"""
from .pipeline import Result, Suggestion, process, tag

__all__ = ["process", "tag", "load_tagger", "Result", "Suggestion"]
__version__ = "0.1.0"


def load_tagger(model_id: str | None = None):
    """FashionCLIP by default (downloads ~600 MB from Hugging Face once)."""
    from .tagger import DEFAULT_MODEL, ClipTagger
    return ClipTagger(model_id or DEFAULT_MODEL)
